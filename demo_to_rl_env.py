from __future__ import annotations

"""
这个文件把原始 pygame 物理 demo 包装成一个 Gymnasium 强化学习环境。



1. observation（观测）
   环境每一步会给智能体一个长度为 5 的数组：
   [vx, vy, m, ux_obs, uy_obs]

   - vx, vy：智能体当前速度
   - m：当前是否“看见”食物，1 表示看见，0 表示看不见
   - ux_obs, uy_obs：看见时，食物方向的单位向量

   注意：
   这里不是食物的真实当前位置，而是“按照有限光速传播后，现在能观测到的方向”。

2. action（动作）
   策略网络输出一个二维向量 [ax_raw, ay_raw]，每个分量范围是 [-1, 1]。
   这个向量先被投影到单位圆内，再乘以当前最大加速度，变成真正施加到智能体上的加速度。

   所以动作不是“下一时刻速度”，也不是“下一时刻位置”，
   而是“控制输入”，即加速度命令。

3. step（状态推进）
   每一步都遵循：
   动作 -> 加速度 -> 速度变化 -> 位置变化 -> 更新可见性 -> 计算奖励

   这点非常重要，因为强化学习里经常会犯错，把动作直接当速度或位移。
"""

import argparse
import math
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from minkowski_worldline_stable import (
    ARENA_RADIUS,
    BASE_PLAYER_RADIUS,
    BASE_POINT_RADIUS,
    BG,
    C,
    FPS,
    HEIGHT,
    HUD,
    PLAYER_ACCEL,
    PLAYER_COLOR,
    POINT_COLOR,
    State,
    WIDTH,
    WorldlineBody,
    advance_body,
    coordinate_velocity_from_proper_velocity,
    draw_player,
    gamma_from_proper_velocity,
    intersect_past_lightcone,
    proper_velocity_from_coordinate_velocity,
    render_visible_point,
    score_scaled_body_scale,
    score_scaled_player_accel,
    score_scaled_point_speed,
    score_scaled_spawn_radius,
    score_scaled_view_scale,
)
from utils import project_to_unit_ball, safe_unit_vector


Vec2 = pygame.math.Vector2


# 核心 RL 环境文件：
# 该文件在不重写原始物理骨架的前提下，把 demo 包装成 Gymnasium 环境，
# 供 SAC 等连续控制算法直接调用。
class RelativisticFoodEnv(gym.Env):
    """
    基于原始 Minkowski worldline demo 构建的 Gymnasium 环境。

    这个环境保留了原 demo 的主要物理思想：
    - 食物和玩家都在二维世界里运动；
    - 食物信号以有限速度传播；
    - 玩家当前能看到的不是食物真实位置，而是过去光锥上的“可见事件”；
    - 玩家吃到食物后，分数增加，最大加速度上限也提高。

    为了适配强化学习，做了这些最小修改：
    - 删除键盘输入，改成策略网络输出二维连续动作；
    - 把全局模拟时间变量改名为 coordinate_time，避免和 proper time 混淆；
    - 食物刷新使用环境自己的随机数生成器，便于 reset(seed=...) 复现结果。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        dt: float = 1.0 / FPS,
        max_steps: int = 1800,
        food_escape_radius: float = ARENA_RADIUS,
        world_radius: float = ARENA_RADIUS * 3.0,
        reward_shaping: bool = False,
        shaping_scale: float = 0.02,
    ) -> None:
        """
        参数说明
        ----------
        render_mode:
            渲染模式。
            - None：不渲染，训练时通常用这个，速度最快
            - "human"：打开 pygame 窗口，肉眼观看
            - "rgb_array"：返回图像数组，常用于录视频或和别的工具对接

        dt:
            每一步的物理时间间隔，单位是“秒”。
            默认是 1 / FPS，也就是约 1/60 秒。
            - 变大：每一步跨得更远，训练更快，但物理更粗糙
            - 变小：物理更细致，但训练更慢

        max_steps:
            一个 episode 最多跑多少步。
            达到这个步数后，本回合会被 truncated（截断）而不是 terminated（异常终止）。
            初学者可以把它理解成“每局游戏的最长时长”。

        food_escape_radius:
            如果食物离玩家太远，超过这个半径，就认为它“跑丢了”，直接刷新一个新的食物。
            这样做是为了避免食物漂得太远，导致训练长时间没有有效交互。

        world_radius:
            如果玩家自己跑得太远，超过这个边界，也会截断当前回合。
            这是一个数值稳定性保护。

        reward_shaping:
            是否启用额外的稠密奖励。
            - False：默认关闭，只保留“吃到食物 +1”的稀疏奖励
            - True：会额外给一点“离食物更近/更远”的辅助奖励

        shaping_scale:
            稠密奖励的缩放系数。
            只有 reward_shaping=True 时才会生效。
            数值越大，辅助奖励影响越明显。
        """
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.render_mode = render_mode
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.food_escape_radius = float(food_escape_radius)
        self.world_radius = float(world_radius)
        self.reward_shaping = bool(reward_shaping)
        self.shaping_scale = float(shaping_scale)

        # 动作是二维连续控制量，范围先规范为 [-1, 1]^2，
        # 后续在 step() 中再投影到单位圆并映射成真实加速度。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 观测定义为 [vx, vy, m, ux_obs, uy_obs]。
        # 其中 m 为可见性 mask，方向是“当前可观测方向”而不是食物真实位置方向。
        self.observation_space = spaces.Box(
            low=np.array([-C, -C, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([C, C, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.coordinate_time = 0.0
        self.step_count = 0
        self.score = 0
        self.player: WorldlineBody | None = None
        self.food: WorldlineBody | None = None
        self._last_true_distance: float | None = None

        self._pygame_ready = False
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None

    def _make_player(self) -> WorldlineBody:
        # 环境内部始终维护真实位置和速度；渲染时“玩家在中心”只是参考系展示方式。
        return WorldlineBody(
            name="agent",
            color=PLAYER_COLOR,
            radius=BASE_PLAYER_RADIUS,
            state=State(Vec2(0.0, 0.0), Vec2(0.0, 0.0), 0.0),
        )

    def _spawn_food(self, origin: Vec2) -> WorldlineBody:
        # 食物刷新规则沿用原 demo：
        # 出生位置与当前分数、视角缩放有关，并且带有随机漂移速度。
        view_scale = score_scaled_view_scale(self.score)
        spawn_min, spawn_max = score_scaled_spawn_radius(self.score, view_scale)

        angle = float(self.np_random.uniform(0.0, math.tau))
        distance = float(self.np_random.uniform(spawn_min, spawn_max))
        position = origin + Vec2(math.cos(angle), math.sin(angle)) * distance

        drift_angle = float(self.np_random.uniform(0.0, math.tau))
        drift_speed = score_scaled_point_speed(self.score)
        drift_velocity = Vec2(math.cos(drift_angle), math.sin(drift_angle)) * drift_speed

        return WorldlineBody(
            name="food",
            color=POINT_COLOR,
            radius=BASE_POINT_RADIUS,
            state=State(position, proper_velocity_from_coordinate_velocity(drift_velocity), 0.0),
        )

    def _assert_state_ready(self) -> tuple[WorldlineBody, WorldlineBody]:
        if self.player is None or self.food is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")
        return self.player, self.food

    def _player_body_scale(self) -> float:
        return score_scaled_body_scale(self.score)

    def _player_radius(self) -> float:
        return BASE_PLAYER_RADIUS * self._player_body_scale()

    def _food_radius(self) -> float:
        return BASE_POINT_RADIUS * self._player_body_scale()

    def _max_acceleration(self) -> float:
        return score_scaled_player_accel(self.score)

    def _observable_food_event(self):
        # 不是直接看食物“当前”位置，而是求食物世界线与玩家过去光锥的交点。
        player, food = self._assert_state_ready()
        observer = player.latest_event()
        return intersect_past_lightcone(food.history, observer)

    def _player_coordinate_velocity(self) -> Vec2:
        player, _ = self._assert_state_ready()
        return coordinate_velocity_from_proper_velocity(player.state.vel)

    def _true_food_distance(self) -> float:
        player, food = self._assert_state_ready()
        return float(player.state.pos.distance_to(food.state.pos))

    def _state_is_finite(self) -> bool:
        player, food = self._assert_state_ready()
        values = [
            self.coordinate_time,
            player.state.pos.x,
            player.state.pos.y,
            player.state.vel.x,
            player.state.vel.y,
            player.state.tau,
            food.state.pos.x,
            food.state.pos.y,
            food.state.vel.x,
            food.state.vel.y,
            food.state.tau,
        ]
        return all(math.isfinite(value) for value in values)

    def _get_obs(self) -> np.ndarray:
        """
        生成当前时刻给策略网络的观测。

        返回值是一个 shape=(5,) 的 numpy 数组：
        [vx, vy, m, ux_obs, uy_obs]

        这里最值得注意的是后 3 个量：
        - m=1：说明食物此刻在“可观测意义下”可见
        - m=0：说明看不见，此时方向自动置零
        - ux_obs, uy_obs：只有在 m=1 时才有意义
        """
        player, _ = self._assert_state_ready()
        coordinate_velocity = self._player_coordinate_velocity()
        observer = player.latest_event()
        visible_event = self._observable_food_event()

        obs = np.zeros(5, dtype=np.float32)
        obs[0] = float(coordinate_velocity.x)
        obs[1] = float(coordinate_velocity.y)

        if visible_event is not None:
            # 只有在过去光锥能看到食物信号时，mask 才置为 1，
            # 并返回可见事件相对玩家的单位方向向量。
            relative = visible_event.vec() - observer.vec()
            unit_direction, norm = safe_unit_vector((relative.x, relative.y))
            if norm > 0.0:
                obs[2] = 1.0
                obs[3:] = unit_direction

        return obs

    def _get_info(self, *, ate_food: bool = False, respawned: bool = False) -> dict[str, Any]:
        # info 不直接给策略网络使用，主要用于调试、打印日志和分析训练过程。
        coordinate_velocity = self._player_coordinate_velocity()
        obs = self._get_obs()
        player, _ = self._assert_state_ready()
        return {
            "score": self.score,
            "ate_food": ate_food,
            "food_respawned": respawned,
            "coordinate_time": self.coordinate_time,
            "proper_time": float(player.state.tau),
            "coordinate_speed": float(coordinate_velocity.length()),
            "gamma": float(gamma_from_proper_velocity(player.state.vel)),
            "accel_limit": self._max_acceleration(),
            "food_visible": bool(obs[2] > 0.5),
            "observed_direction": obs[3:].copy(),
            "food_distance": self._true_food_distance(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # reset() 重置时间、分数和实体状态，并重新记录世界线起点。
        #
        # 返回:
        # - obs: 初始观测
        # - info: 附加信息（例如当前分数、速度、是否看见食物等）
        super().reset(seed=seed)

        self.coordinate_time = 0.0
        self.step_count = 0
        self.score = 0

        self.player = self._make_player()
        self.food = self._spawn_food(self.player.state.pos)

        self.player.record(self.coordinate_time)
        self.food.record(self.coordinate_time)
        self._last_true_distance = self._true_food_distance()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行一步环境更新。

        参数
        ----
        action:
            策略网络输出的二维动作，理论范围是 [-1, 1]^2。
            这个动作不是速度，也不是位移，而是“想施加的加速度方向和强度”。

        返回
        ----
        obs:
            下一时刻观测
        reward:
            当前这一步得到的奖励
        terminated:
            是否因为异常/非法数值等原因终止
        truncated:
            是否因为超步数/超边界等原因被截断
        info:
            调试信息字典
        """
        self._assert_state_ready()

        # 先把策略输出投影到单位圆，再按当前分数映射到真实最大加速度。
        projected_action = project_to_unit_ball(action)
        max_acceleration = self._max_acceleration()
        applied_acceleration = Vec2(float(projected_action[0]), float(projected_action[1])) * max_acceleration

        start_t = self.coordinate_time
        end_t = start_t + self.dt

        # 关键约束：
        # 动作代表“控制输入”，真正的状态推进仍由原 demo 的 advance_body() 负责，
        # 也就是加速度 -> 速度 -> 位置，而不是把动作直接当速度或位移。
        advance_body(self.player, start_t, end_t, applied_acceleration)
        advance_body(self.food, start_t, end_t)
        self.coordinate_time = end_t
        self.step_count += 1

        reward = 0.0
        ate_food = False
        respawned = False
        termination_reason: str | None = None
        truncation_reason: str | None = None

        hit_radius = self._player_radius() + self._food_radius()
        true_distance = self._true_food_distance()
        escaped = true_distance > self.food_escape_radius

        # 默认稀疏奖励：吃到食物 +1，其余时间 0。
        if true_distance <= hit_radius:
            reward += 1.0
            ate_food = True
            respawned = True
            self.score += 1
        elif escaped:
            respawned = True

        # shaping 默认关闭；这里只有在显式开启时才添加一个轻量距离差奖励。
        if self.reward_shaping and self._last_true_distance is not None and not respawned:
            distance_delta = self._last_true_distance - true_distance
            reward += self.shaping_scale * float(np.clip(distance_delta / self.food_escape_radius, -1.0, 1.0))

        if respawned:
            player, _ = self._assert_state_ready()
            self.food = self._spawn_food(player.state.pos)
            self.food.record(self.coordinate_time)

        terminated = False
        truncated = False

        # 训练稳定性保护：
        # 非法数值直接终止，超边界或超步数则截断。
        if not self._state_is_finite():
            terminated = True
            termination_reason = "non_finite_state"
        elif self.player.state.pos.length() > self.world_radius:
            truncated = True
            truncation_reason = "player_out_of_bounds"
        elif self.step_count >= self.max_steps:
            truncated = True
            truncation_reason = "max_steps"

        self._last_true_distance = self._true_food_distance()

        obs = self._get_obs()
        info = self._get_info(ate_food=ate_food, respawned=respawned)
        info["raw_action"] = np.asarray(action, dtype=np.float32).copy()
        info["projected_action"] = projected_action.copy()
        info["applied_acceleration"] = np.array(
            [applied_acceleration.x, applied_acceleration.y], dtype=np.float32
        )

        if termination_reason is not None:
            info["termination_reason"] = termination_reason
        if truncation_reason is not None:
            info["truncation_reason"] = truncation_reason

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _init_rendering(self) -> None:
        # 延迟初始化 pygame，避免无渲染模式下也创建窗口。
        if self._pygame_ready:
            return

        pygame.init()
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Relativistic Food RL")
            self._clock = pygame.time.Clock()
        else:
            self._window = pygame.Surface((WIDTH, HEIGHT))

        self._font = pygame.font.Font(None, 24)
        self._small_font = pygame.font.Font(None, 18)
        self._pygame_ready = True

    def _draw_hud(self, surface: pygame.Surface) -> None:
        if self._font is None or self._small_font is None:
            raise RuntimeError("Fonts are not initialized")

        # HUD 仅用于调试和观察训练后的行为，不参与策略输入。
        accel_scale = self._max_acceleration() / PLAYER_ACCEL
        obs = self._get_obs()

        lines = [
            self._font.render(f"score {self.score}", True, HUD),
            self._font.render(f"thrust {accel_scale:0.2f}x", True, HUD),
            self._font.render(f"speed {self._player_coordinate_velocity().length():0.1f}", True, HUD),
            self._small_font.render(
                f"visible {int(obs[2])}  dir ({obs[3]:+0.2f}, {obs[4]:+0.2f})",
                True,
                HUD,
            ),
        ]

        for idx, line in enumerate(lines):
            surface.blit(line, (18, 16 + idx * 28))

    def render(self) -> np.ndarray | None:
        """
        渲染当前画面。

        - render_mode="human" 时：直接显示在 pygame 窗口中
        - render_mode="rgb_array" 时：返回图像数组
        - render_mode=None 时：不做任何事
        """
        if self.render_mode is None:
            return None

        self._assert_state_ready()
        self._init_rendering()
        if self._window is None:
            raise RuntimeError("Render surface was not created")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_ready = False
                self._window = None
                return None

        player, food = self._assert_state_ready()
        observer = player.latest_event()
        point_radius = self._food_radius()
        player_radius = self._player_radius()
        pulse = 0.5 + 0.5 * math.sin(self.coordinate_time * 2.4)
        view_scale = score_scaled_view_scale(self.score)

        self._window.fill(BG)
        render_visible_point(self._window, food, observer, view_scale, point_radius)
        draw_player(self._window, player_radius, pulse)
        self._draw_hud(self._window)

        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None

        pixels = pygame.surfarray.array3d(self._window)
        return np.transpose(pixels, (1, 0, 2))

    def close(self) -> None:
        # 统一释放 pygame 资源，便于训练/评估脚本多次创建环境。
        if self._pygame_ready:
            pygame.quit()
        self._pygame_ready = False
        self._window = None
        self._clock = None
        self._font = None
        self._small_font = None


def random_rollout(steps: int = 600, render: bool = True, seed: int = 0) -> None:
    # 这个函数用于环境自检：不用模型，直接随机采样动作跑若干步。
    #
    # 参数:
    # - steps: 最多随机走多少步
    # - render: 是否显示窗口
    # - seed: 随机种子，便于复现实验
    render_mode = "human" if render else None
    env = RelativisticFoodEnv(render_mode=render_mode)
    try:
        obs, info = env.reset(seed=seed)
        episode_reward = 0.0
        for _ in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        print("Random rollout summary")
        print(f"  reward: {episode_reward:.2f}")
        print(f"  score:  {info['score']}")
        print(f"  obs:    {obs}")
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    # 允许直接运行本文件来做 smoke test。
    parser = argparse.ArgumentParser(description="Smoke-test the relativistic RL environment.")
    parser.add_argument("--steps", type=int, default=600, help="Number of random-policy steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reset().")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable pygame rendering during the random-policy rollout.",
    )
    return parser


def main() -> None:
    # 直接执行 demo_to_rl_env.py 时，默认进入随机动作回放模式。
    args = build_arg_parser().parse_args()
    random_rollout(steps=args.steps, render=not args.no_render, seed=args.seed)


if __name__ == "__main__":
    main()
