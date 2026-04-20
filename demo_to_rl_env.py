from __future__ import annotations  # 允许在类型注解里直接使用尚未定义的类名，并推迟注解求值。

"""
把原始 pygame 相对论贪吃蛇 demo 包装成一个 Gymnasium 环境。

这个文件的目标不是重写物理，而是尽量少改原始 demo 的核心机制，
只把它整理成可以被 RL 算法调用的接口。

当前 observation 定义为：
    obs = [m, x_obs, y_obs]

其中：
    - m: 当前是否观测到食物，能观测到时为 1，否则为 0。
    - x_obs, y_obs: 当前“可见食物事件”在实验室系里的空间坐标。

这里特别强调两点：
    - 这不是食物的真实当前坐标，而是过去光锥上可见事件的坐标。
    - 这不是单位方向向量，也不是蛇瞬时参考系里的位置。

动作仍然是二维连续控制，物理更新仍然走原 demo 的相对论动力学。
"""

import argparse  # 负责解析命令行参数，便于做 smoke test。
import math  # 提供三角函数、有限性判断和 tau 等数学工具。
from typing import Any  # 用于宽松的类型标注，例如 info 字典。

import gymnasium as gym  # Gymnasium 提供标准 RL 环境基类。
import numpy as np  # NumPy 用于构造 observation、action 和数值数组。
import pygame  # pygame 负责 2D 向量、窗口、绘制和事件循环。
from gymnasium import spaces  # spaces 用于定义 action_space 和 observation_space。

# 下面这一组导入全部直接来自原始 demo 文件。
# 这样可以最大限度复用原工程的物理、渲染和分数缩放逻辑，而不是在 RL 环境里重写一套。
from 相对论贪吃蛇 import (
    ARENA_RADIUS,  # 原 demo 的基础场地半径。
    BASE_PLAYER_RADIUS,  # 玩家在基础难度下的半径。
    BASE_POINT_RADIUS,  # 食物在基础难度下的半径。
    BG,  # 背景颜色。
    C,  # 光速常量。
    Event,  # 世界线历史中的单个事件。
    FPS,  # 原 demo 使用的目标帧率。
    HEIGHT,  # 渲染窗口高度。
    HUD,  # HUD 文本颜色。
    PLAYER_ACCEL,  # 原 demo 的基础玩家加速度。
    PLAYER_COLOR,  # 玩家颜色。
    POINT_COLOR,  # 食物颜色。
    State,  # 物体的当前状态，包含位置、proper velocity 和 proper time。
    WIDTH,  # 渲染窗口宽度。
    WorldlineBody,  # 带世界线历史记录的运动物体。
    advance_body,  # 原 demo 的相对论动力学推进函数。
    coordinate_velocity_from_proper_velocity,  # proper velocity -> 坐标速度。
    draw_player,  # 原 demo 的玩家绘制函数。
    gamma_from_proper_velocity,  # 由 proper velocity 计算 gamma 因子。
    intersect_past_lightcone,  # 求过去光锥与食物世界线的可见交点。
    proper_velocity_from_coordinate_velocity,  # 坐标速度 -> proper velocity。
    render_visible_point,  # 按过去光锥结果渲染“看见的食物”。
    score_scaled_body_scale,  # 分数越高，身体尺寸缩放越大。
    score_scaled_player_accel,  # 分数越高，可用加速度上限会变化。
    score_scaled_point_speed,  # 分数越高，食物漂移速度会变化。
    score_scaled_spawn_radius,  # 分数越高，食物生成距离范围会变化。
    score_scaled_view_scale,  # 分数越高，视图尺度会变化。
    transform_event_to_player_frame,  # 把事件变到玩家瞬时参考系；这里只用于 HUD/debug。
)
from utils import project_to_unit_ball  # 把二维动作投影到单位圆盘内，防止控制过强。


# 直接复用 pygame 的二维向量类型，后面位置、速度、加速度都统一用它。
Vec2 = pygame.math.Vector2


class RelativisticFoodEnv(gym.Env):
    """
    对原始相对论贪吃蛇 demo 的最小 RL 包装。

    保留的核心逻辑：
    - 玩家和食物都在二维空间里运动。
    - 食物是否可见，仍然通过过去光锥和食物世界线求交决定。
    - 动作控制的是加速度，而不是直接控制速度或位置。
    - 分数仍然影响身体尺度、生成半径、视图比例和加速度上限。

    RL 化以后增加的只是：
    - 用连续动作替代键盘输入。
    - 用 reset/step/render 暴露标准 Gymnasium 接口。
    - 用环境自己的随机数生成器保证 reset 可复现。
    """

    # 告诉 Gymnasium 这个环境支持两种渲染模式，并声明人类模式下的目标帧率。
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,  # 渲染模式，None 表示不渲染。
        dt: float = 1.0 / FPS,  # 每个 step 对应的实验室系时间步长。
        max_steps: int = 5000,  # 单局最多推进多少个 step。
        food_escape_radius: float = ARENA_RADIUS,  # 食物离玩家太远时触发重生的半径。
        world_radius: float = ARENA_RADIUS * 3.0,  # 玩家允许活动的安全范围。
        reward_shaping: bool = True,  # 是否启用基于距离变化的 shaping reward。
        shaping_scale: float = 0.5,  # shaping reward 的强度系数。
    ) -> None:
        super().__init__()  # 先初始化 Gymnasium 基类。

        # 如果 render_mode 不是 None，就必须出现在 metadata 声明过的模式里。
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        # 时间步长必须为正，否则动力学推进没有意义。
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # 单局最大步数至少要大于 0，否则环境一开始就该截断。
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        # 保存构造参数，供后续 step、render 和 info 使用。
        self.render_mode = render_mode
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.food_escape_radius = float(food_escape_radius)
        self.world_radius = float(world_radius)
        self.reward_shaping = bool(reward_shaping)
        self.shaping_scale = float(shaping_scale)

        # 动作空间仍然保持二维连续控制，每个分量都在 [-1, 1]。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation 现在只表达三件事：
        # 1) 当前是否看见食物；
        # 2) 看见时，对应可见事件在实验室系里的 x 坐标；
        # 3) 看见时，对应可见事件在实验室系里的 y 坐标。
        #
        # x_obs/y_obs 的范围按“玩家世界边界 + 食物逃逸边界”给一个保守上界，
        # 这样不需要做额外归一化，也避免大改现有代码。
        observable_position_limit = self.world_radius + self.food_escape_radius

        # observation_space 的第 0 维是可见性标记，只取 [0, 1]；
        # 后两维是观测到的位置坐标，因此允许在正负边界间变化。
        self.observation_space = spaces.Box(
            low=np.array([0.0, -observable_position_limit, -observable_position_limit], dtype=np.float32),
            high=np.array([1.0, observable_position_limit, observable_position_limit], dtype=np.float32),
            dtype=np.float32,
        )

        # coordinate_time 表示实验室系下已经推进了多少时间。
        self.coordinate_time = 0.0

        # step_count 记录当前 episode 已经走了多少步。
        self.step_count = 0

        # score 记录当前已经吃到多少次食物。
        self.score = 0

        # player 指向玩家这个世界线物体；reset 之前先置空。
        self.player: WorldlineBody | None = None

        # food 指向当前食物这个世界线物体；reset 之前也先置空。
        self.food: WorldlineBody | None = None

        # _last_true_distance 用于 reward shaping，记录上一步的真实距离。
        self._last_true_distance: float | None = None

        # 下面这些字段都只和渲染有关，因此在真正需要前不初始化。
        self._pygame_ready = False  # 标记 pygame 资源是否已经创建完毕。
        self._window: pygame.Surface | None = None  # 人类模式下是窗口，rgb_array 模式下是离屏 surface。
        self._clock: pygame.time.Clock | None = None  # 人类模式下用于限帧。
        self._font: pygame.font.Font | None = None  # HUD 主字体。
        self._small_font: pygame.font.Font | None = None  # HUD 次级字体。

    def _make_player(self) -> WorldlineBody:
        """创建玩家初始状态。"""

        # 玩家初始时位于原点，proper velocity 为 0，proper time 为 0。
        return WorldlineBody(
            name="agent",  # 物体名字仅用于标识。
            color=PLAYER_COLOR,  # 玩家颜色沿用原 demo。
            radius=BASE_PLAYER_RADIUS,  # 半径使用基础玩家半径。
            state=State(Vec2(0.0, 0.0), Vec2(0.0, 0.0), 0.0),  # 初始位置、速度、固有时。
        )

    def _spawn_food(self, origin: Vec2) -> WorldlineBody:
        """围绕给定 origin 生成一个新的食物。"""

        # 先根据当前分数得到视图缩放；原 demo 的生成半径逻辑依赖它。
        view_scale = score_scaled_view_scale(self.score)

        # 再根据分数和视图尺度拿到允许的生成距离区间。
        spawn_min, spawn_max = score_scaled_spawn_radius(self.score, view_scale)

        # 随机采样一个极角，决定食物生成方向。
        angle = float(self.np_random.uniform(0.0, math.tau))

        # 再随机采样一个距离，决定食物离 origin 有多远。
        distance = float(self.np_random.uniform(spawn_min, spawn_max))

        # 把极坐标转换成笛卡尔坐标，得到食物生成位置。
        position = origin + Vec2(math.cos(angle), math.sin(angle)) * distance

        # 再采样一个漂移方向，用于食物自身的初始运动方向。
        drift_angle = float(self.np_random.uniform(0.0, math.tau))

        # 食物漂移速度大小仍然完全复用原 demo 的分数缩放逻辑。
        drift_speed = score_scaled_point_speed(self.score)

        # 生成实验室系下的食物漂移速度向量。
        drift_velocity = Vec2(math.cos(drift_angle), math.sin(drift_angle)) * drift_speed

        # WorldlineBody 内部保存的是 proper velocity，因此这里要先做一次速度转换。
        return WorldlineBody(
            name="food",  # 物体名字标为 food。
            color=POINT_COLOR,  # 颜色沿用原 demo 的食物颜色。
            radius=BASE_POINT_RADIUS,  # 食物基础半径。
            state=State(position, proper_velocity_from_coordinate_velocity(drift_velocity), 0.0),
        )

    def _assert_state_ready(self) -> tuple[WorldlineBody, WorldlineBody]:
        """确保 player 和 food 都已经在 reset() 后初始化。"""

        # 如果任一对象为空，说明调用顺序有误，必须先 reset。
        if self.player is None or self.food is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")

        # 状态合法时直接返回，方便调用方同时拿到 player 和 food。
        return self.player, self.food

    def _player_body_scale(self) -> float:
        """按当前分数查询玩家尺寸缩放倍率。"""

        return score_scaled_body_scale(self.score)  # 完全复用原 demo 的分数缩放曲线。

    def _player_radius(self) -> float:
        """返回当前分数下玩家的实际半径。"""

        return BASE_PLAYER_RADIUS * self._player_body_scale()  # 基础半径乘以缩放倍率。

    def _food_radius(self) -> float:
        """返回当前分数下食物的实际半径。"""

        return BASE_POINT_RADIUS * self._player_body_scale()  # 食物和玩家共用同一套缩放倍率。

    def _max_acceleration(self) -> float:
        """返回当前分数下允许的最大控制加速度。"""

        return score_scaled_player_accel(self.score)  # 直接复用原 demo 的难度曲线。

    def _action_to_applied_acceleration(self, action: np.ndarray) -> tuple[np.ndarray, Vec2]:
        """
        把策略网络输出的动作变成真正用于物理推进的二维加速度。
        """

        # 先把动作投影到单位圆盘内，避免对角线方向的动作模长超过 1。
        projected_action = project_to_unit_ball(action)

        # 再根据当前分数拿到可用的加速度上限。
        accel_limit = self._max_acceleration()

        # 最终控制量 = 单位圆盘内的方向/幅值 * 当前允许的最大加速度。
        applied_acceleration = Vec2(float(projected_action[0]), float(projected_action[1])) * accel_limit

        # 把投影后的动作和真实施加的加速度一起返回，前者便于写入 info 调试。
        return projected_action, applied_acceleration

    def _advance_relativistic_dynamics(self, applied_acceleration: Vec2) -> None:
        """
        用原始 demo 的相对论动力学更新玩家和食物。
        """

        # 当前 step 的起始实验室系时间。
        start_t = self.coordinate_time

        # 当前 step 的结束实验室系时间。
        end_t = start_t + self.dt

        # 玩家带控制输入推进；这里完全复用原 demo 的 advance_body。
        advance_body(self.player, start_t, end_t, applied_acceleration)

        # 食物没有外部控制输入，只按自身状态自然推进。
        advance_body(self.food, start_t, end_t)

        # 推进结束后，把全局时间更新到 step 终点。
        self.coordinate_time = end_t

        # 同时累加 step 计数。
        self.step_count += 1

    def _observable_food_event(self) -> Event | None:
        """
        求当前玩家过去光锥与食物世界线的交点，也就是“当前能看到的食物事件”。
        """

        # 先拿到已经初始化好的 player 和 food。
        player, food = self._assert_state_ready()

        # 观测者事件取玩家世界线上的最新事件。
        observer = player.latest_event()

        # 用过去光锥与食物历史求交；若没有交点，则当前不可见。
        return intersect_past_lightcone(food.history, observer)

    def _observable_food_position(self) -> tuple[float, float] | None:
        """
        返回当前可见食物事件在实验室系里的二维位置。

        这是 observation 真正使用的空间信息来源。
        """

        # 先求出当前可见事件。
        visible_event = self._observable_food_event()

        # 如果没有可见事件，说明当前看不到食物，直接返回 None。
        if visible_event is None:
            return None

        # 只取事件本身的实验室系 x/y 坐标，不做单位化，也不做参考系变换。
        return float(visible_event.x), float(visible_event.y)

    def _player_coordinate_velocity(self) -> Vec2:
        """
        把玩家内部保存的 proper velocity 转成更易读的实验室系坐标速度。
        """

        # 取出玩家当前状态。
        player, _ = self._assert_state_ready()

        # 转成 coordinate velocity；这个值主要用于 HUD 和 info 调试。
        return coordinate_velocity_from_proper_velocity(player.state.vel)

    def _observable_food_in_player_frame(self) -> tuple[float, Vec2] | None:
        """
        仅供 HUD / debug 使用：把当前可见食物事件变换到玩家瞬时参考系。

        注意：
        这个函数不参与 observation 构造；
        observation 只使用实验室系下 visible_event 的 x/y 坐标。
        """

        # 先确保 player 和 food 都存在。
        player, _ = self._assert_state_ready()

        # 取当前可见事件；若不可见则后续也无需计算。
        visible_event = self._observable_food_event()

        # 不可见时直接返回 None，方便 HUD 用 "--" 显示。
        if visible_event is None:
            return None

        # 观测者事件仍然取玩家世界线最新点。
        observer = player.latest_event()

        # 观测者速度要用坐标速度形式传给原 demo 的 Lorentz 变换函数。
        observer_velocity = self._player_coordinate_velocity()

        # 返回 (t', r')，其中 t' 是玩家系时间，r' 是玩家系空间位置。
        return transform_event_to_player_frame(visible_event, observer, observer_velocity)

    def _true_food_distance(self) -> float:
        """返回玩家和食物在当前真实状态下的欧式距离。"""

        # 拿到当前真实状态下的 player 和 food。
        player, food = self._assert_state_ready()

        # 这里比较的是两者“现在”的位置，不是过去光锥上的可见事件位置。
        return float(player.state.pos.distance_to(food.state.pos))

    def _state_is_finite(self) -> bool:
        """检查关键状态量里是否出现 NaN 或 inf。"""

        # 先把 player 和 food 取出来，下面统一检查。
        player, food = self._assert_state_ready()

        # 把所有关键标量整理成一个列表，便于统一调用 math.isfinite。
        values = [
            self.coordinate_time,  # 当前实验室系时间。
            player.state.pos.x,  # 玩家 x 坐标。
            player.state.pos.y,  # 玩家 y 坐标。
            player.state.vel.x,  # 玩家 proper velocity 的 x 分量。
            player.state.vel.y,  # 玩家 proper velocity 的 y 分量。
            player.state.tau,  # 玩家固有时。
            food.state.pos.x,  # 食物 x 坐标。
            food.state.pos.y,  # 食物 y 坐标。
            food.state.vel.x,  # 食物 proper velocity 的 x 分量。
            food.state.vel.y,  # 食物 proper velocity 的 y 分量。
            food.state.tau,  # 食物固有时。
        ]

        # 所有值都有限时返回 True，只要有一个异常就返回 False。
        return all(math.isfinite(value) for value in values)

    def _get_obs(self) -> np.ndarray:
        """
        构造当前 observation。

        observation 的唯一有效定义就是：
            [m, x_obs, y_obs]

        其中：
        - m = 1 表示当前能看到食物，否则为 0；
        - x_obs, y_obs 是可见食物事件在实验室系里的坐标；
        - 当前不可见时，统一返回 [0, 0, 0]。
        """

        # 先默认构造“看不见食物”的 observation。
        obs = np.zeros(3, dtype=np.float32)

        # 尝试取得当前可见事件的实验室系位置。
        observed_position = self._observable_food_position()

        # 如果返回 None，就保持默认的 [0, 0, 0]。
        if observed_position is None:
            return obs

        # 否则把实验室系下的可见事件坐标拆开。
        x_obs, y_obs = observed_position

        # 第 0 维置为 1，表示当前能看到食物。
        obs[0] = 1.0

        # 第 1 维写入可见事件的 x 坐标。
        obs[1] = x_obs

        # 第 2 维写入可见事件的 y 坐标。
        obs[2] = y_obs

        # 返回最终 observation。
        return obs

    def _get_info(self, *, ate_food: bool = False, respawned: bool = False) -> dict[str, Any]:
        """
        生成调试信息字典。

        info 不参与策略输入，但方便训练日志、可视化和人工排查。
        """

        # 先拿到玩家当前坐标速度，便于记录速度相关调试量。
        coordinate_velocity = self._player_coordinate_velocity()

        # 再拿一次玩家系下的可见事件，用来给 HUD/debug 提供 t'。
        observed_in_player_frame = self._observable_food_in_player_frame()

        # observation 本身也读一次，避免重复拼字段逻辑。
        obs = self._get_obs()

        # 玩家对象本身要用于读取 proper time 和 gamma。
        player, _ = self._assert_state_ready()

        # 直接返回调试字段字典；下面每一项都加了明确语义说明。
        return {
            "score": self.score,  # 当前累计得分。
            "ate_food": ate_food,  # 这一 step 是否真的吃到了食物。
            "food_respawned": respawned,  # 这一 step 后食物是否被重生。
            "coordinate_time": self.coordinate_time,  # 当前实验室系时间。
            "proper_time": float(player.state.tau),  # 玩家当前固有时。
            "observation_frame": "lab_visible_event_position",  # observation 的参考系说明。
            "coordinate_speed": float(coordinate_velocity.length()),  # 玩家实验室系速度大小。
            "gamma": float(gamma_from_proper_velocity(player.state.vel)),  # 玩家当前 gamma。
            "accel_limit": self._max_acceleration(),  # 当前允许的加速度上限。
            "food_visible": bool(obs[0] > 0.5),  # 是否可见；和 observation 第 0 维一致。
            "observed_position": obs[1:].copy(),  # observation 里的位置部分复制出来，便于日志记录。
            "observed_t_prime": None if observed_in_player_frame is None else float(observed_in_player_frame[0]),  # HUD/debug 用的玩家系时间。
            "food_distance": self._true_food_distance(),  # 玩家和食物“真实当前状态”的距离。
        }

    def reset(
        self,
        *,
        seed: int | None = None,  # reset 时可传随机种子，保证可复现。
        options: dict[str, Any] | None = None,  # 预留给 Gymnasium 的可选参数。
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境到初始状态。"""

        # 先调用父类 reset，让 Gymnasium 正确初始化 self.np_random。
        super().reset(seed=seed)

        # 把实验室系时间归零。
        self.coordinate_time = 0.0

        # 把 step 计数归零。
        self.step_count = 0

        # 把分数归零。
        self.score = 0

        # 创建新的玩家对象。
        self.player = self._make_player()

        # 以玩家当前位置为中心生成新的食物。
        self.food = self._spawn_food(self.player.state.pos)

        # 在 t=0 时刻把玩家初始状态记录进世界线历史。
        self.player.record(self.coordinate_time)

        # 同样把食物初始状态记录进世界线历史。
        self.food.record(self.coordinate_time)

        # 初始化上一时刻真实距离，供后续 shaping reward 使用。
        self._last_true_distance = self._true_food_distance()

        # 如果是 human 模式，reset 后立即渲染首帧，便于看到初始状态。
        if self.render_mode == "human":
            self.render()

        # 返回初始 observation 和对应 info。
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """推进环境一个时间步。"""

        # 先确保 reset() 已经调用过。
        self._assert_state_ready()

        # 把策略输出动作转换成真实控制输入。
        projected_action, applied_acceleration = self._action_to_applied_acceleration(action)

        # 用原 demo 的相对论动力学推进一帧。
        self._advance_relativistic_dynamics(applied_acceleration)

        # 每步 reward 默认从 0 开始累计。
        reward = 0.0

        # ate_food 用来记录这一帧是否吃到了食物。
        ate_food = False

        # respawned 用来记录这一帧结束后是否发生了食物重生。
        respawned = False

        # termination_reason 用于写入 info，说明真正终止的原因。
        termination_reason: str | None = None

        # truncation_reason 用于写入 info，说明截断的原因。
        truncation_reason: str | None = None

        # 吃到食物的判定半径 = 当前玩家半径 + 当前食物半径。
        hit_radius = self._player_radius() + self._food_radius()

        # 计算玩家与食物当前真实状态下的距离。
        true_distance = self._true_food_distance()

        # 如果距离超过逃逸半径，就认为食物跑太远，需要重生。
        escaped = true_distance > self.food_escape_radius

        # 真正吃到食物时，给 +1 reward，并且分数增加、食物重生。
        if true_distance <= hit_radius:
            reward += 1.0
            ate_food = True
            respawned = True
            self.score += 1

        # 没吃到但跑远了，也要重生，只是不加分。
        elif escaped:
            respawned = True

        # 如果启用了 shaping，并且这一步没有重生，就用距离变化给一点辅助 reward。
        if self.reward_shaping and self._last_true_distance is not None and not respawned:
            # 距离减少为正，距离增大为负。
            distance_delta = self._last_true_distance - true_distance

            # 把距离变化按逃逸半径归一化并截断到 [-1, 1]，再乘 shaping_scale。
            reward += self.shaping_scale * float(np.clip(distance_delta / self.food_escape_radius, -1.0, 1.0))

        # 如果这一帧触发了重生，就在玩家当前位置附近生成一个新食物。
        if respawned:
            player, _ = self._assert_state_ready()
            self.food = self._spawn_food(player.state.pos)
            self.food.record(self.coordinate_time)

        # 默认先认为没有终止。
        terminated = False

        # 默认先认为没有截断。
        truncated = False

        # 如果状态里出现 NaN/inf，认为物理数值失稳，直接终止。
        if not self._state_is_finite():
            terminated = True
            termination_reason = "non_finite_state"

        # 如果玩家跑出了允许世界边界，就做截断而不是终止。
        elif self.player.state.pos.length() > self.world_radius:
            truncated = True
            truncation_reason = "player_out_of_bounds"

        # 如果步数超过上限，也做截断。
        elif self.step_count >= self.max_steps:
            truncated = True
            truncation_reason = "max_steps"

        # 无论是否重生，都把“上一帧真实距离”更新成当前最新距离。
        self._last_true_distance = self._true_food_distance()

        # 生成本步结束后的 observation。
        obs = self._get_obs()

        # 生成本步结束后的 info。
        info = self._get_info(ate_food=ate_food, respawned=respawned)

        # 原始动作也记进 info，便于调试策略输出。
        info["raw_action"] = np.asarray(action, dtype=np.float32).copy()

        # 投影后的动作也记进去，方便比较策略输出和真实控制输入差异。
        info["projected_action"] = projected_action.copy()

        # 真实施加的加速度向量也记进 info。
        info["applied_acceleration"] = np.array(
            [applied_acceleration.x, applied_acceleration.y], dtype=np.float32
        )

        # 如果发生真正终止，就在 info 里记录原因。
        if termination_reason is not None:
            info["termination_reason"] = termination_reason

        # 如果发生截断，也在 info 里记录原因。
        if truncation_reason is not None:
            info["truncation_reason"] = truncation_reason

        # human 模式下，每一步结束后都刷新一帧画面。
        if self.render_mode == "human":
            self.render()

        # 按 Gymnasium 约定顺序返回 obs、reward、terminated、truncated、info。
        return obs, float(reward), terminated, truncated, info

    def _init_rendering(self) -> None:
        """按需初始化 pygame 渲染资源。"""

        # 如果已经初始化过，就不要重复创建窗口和字体。
        if self._pygame_ready:
            return

        # 初始化 pygame 全局资源。
        pygame.init()

        # human 模式下创建真实窗口，并准备 clock 做限帧。
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Relativistic Food RL")
            self._clock = pygame.time.Clock()

        # rgb_array 模式下不需要真实窗口，只要离屏 surface 即可。
        else:
            self._window = pygame.Surface((WIDTH, HEIGHT))

        # 创建 HUD 主字体。
        self._font = pygame.font.Font(None, 24)

        # 创建 HUD 小字体。
        self._small_font = pygame.font.Font(None, 18)

        # 标记 pygame 资源已就绪。
        self._pygame_ready = True

    def _draw_hud(self, surface: pygame.Surface) -> None:
        """在画面左上角绘制调试 HUD。"""

        # 如果字体尚未初始化，说明调用顺序有误。
        if self._font is None or self._small_font is None:
            raise RuntimeError("Fonts are not initialized")

        # 当前加速度上限相对于基础 PLAYER_ACCEL 的倍率。
        accel_scale = self._max_acceleration() / PLAYER_ACCEL

        # 玩家当前实验室系速度大小。
        coordinate_speed = self._player_coordinate_velocity().length()

        # 取玩家对象，用于计算 gamma。
        player, _ = self._assert_state_ready()

        # gamma 因子直接由玩家当前 proper velocity 计算。
        gamma = gamma_from_proper_velocity(player.state.vel)

        # beta = v / c，用来直观展示速度占光速的比例。
        beta = coordinate_speed / C

        # 这里仍然保留一次玩家系可见事件计算，只是为了在 HUD 显示 t'。
        observed_in_player_frame = self._observable_food_in_player_frame()

        # 没有可见事件时，用字符串 "--" 占位；有的话显示三位小数的 t'。
        tprime_value = "--" if observed_in_player_frame is None else f"{observed_in_player_frame[0]:0.3f}"

        # 直接复用 _get_obs()，这样 HUD 和实际 observation 始终保持一致。
        obs = self._get_obs()

        # 逐行构造 HUD 文本 surface。
        lines = [
            self._font.render(f"score {self.score}", True, HUD),  # 当前得分。
            self._font.render(f"thrust {accel_scale:0.2f}x", True, HUD),  # 当前控制加速度倍率。
            self._font.render(f"speed {coordinate_speed:0.1f}", True, HUD),  # 当前坐标速度大小。
            self._font.render(f"gamma {gamma:0.3f}  v/c {beta:0.3f}", True, HUD),  # gamma 和 beta。
            self._small_font.render(
                f"visible {int(obs[0])}  t' {tprime_value}  pos ({obs[1]:+0.1f}, {obs[2]:+0.1f})",
                True,
                HUD,
            ),  # 当前 observation 中的可见性与观测位置。
        ]

        # 按固定纵向间隔把每一行文本 blit 到左上角。
        for idx, line in enumerate(lines):
            surface.blit(line, (18, 16 + idx * 28))

    def render(self) -> np.ndarray | None:
        """渲染当前环境状态。"""

        # 如果用户根本没请求渲染，就直接返回 None。
        if self.render_mode is None:
            return None

        # 先确保状态已初始化。
        self._assert_state_ready()

        # 再确保 pygame 资源已经准备好。
        self._init_rendering()

        # 正常情况下 _init_rendering 会创建 window；这里额外做一次安全检查。
        if self._window is None:
            raise RuntimeError("Render surface was not created")

        # human 模式下需要处理窗口事件，避免窗口失去响应。
        for event in pygame.event.get():
            # 如果用户点击关闭窗口，就主动清理 pygame 资源。
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_ready = False
                self._window = None
                return None

        # 取出当前 player 和 food。
        player, food = self._assert_state_ready()

        # 观测者事件用玩家世界线最新事件。
        observer = player.latest_event()

        # 玩家当前坐标速度要传给 render_visible_point，让它按原 demo 逻辑做相对论显示。
        observer_velocity = self._player_coordinate_velocity()

        # 食物绘制半径随当前分数变化。
        point_radius = self._food_radius()

        # 玩家绘制半径也随当前分数变化。
        player_radius = self._player_radius()

        # pulse 用于让玩家有轻微呼吸感动画。
        pulse = 0.5 + 0.5 * math.sin(self.coordinate_time * 2.4)

        # 视图尺度仍然复用原 demo 的分数缩放逻辑。
        view_scale = score_scaled_view_scale(self.score)

        # 每帧先清屏。
        self._window.fill(BG)

        # 按原 demo 逻辑绘制“当前能看到的食物”，这里会自动处理过去光锥可见性。
        render_visible_point(
            self._window,
            food,
            observer,
            observer_velocity,
            view_scale,
            point_radius,
            True,
        )

        # 再绘制玩家本体。
        draw_player(self._window, player_radius, pulse)

        # 最后叠加 HUD。
        self._draw_hud(self._window)

        # human 模式下刷新窗口并按目标 FPS 限帧，不返回像素数组。
        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None

        # rgb_array 模式下把 surface 像素提取成 NumPy 数组。
        pixels = pygame.surfarray.array3d(self._window)

        # pygame 的数组轴顺序是 (W, H, C)，这里转成常见的 (H, W, C)。
        return np.transpose(pixels, (1, 0, 2))

    def close(self) -> None:
        """释放 pygame 相关资源。"""

        # 如果 pygame 已经初始化，就调用 pygame.quit() 做全局清理。
        if self._pygame_ready:
            pygame.quit()

        # 无论之前状态如何，都把内部标记和资源引用复位。
        self._pygame_ready = False
        self._window = None
        self._clock = None
        self._font = None
        self._small_font = None


def random_rollout(steps: int = 600, render: bool = True, seed: int = 0) -> None:
    """用随机策略跑一小段 rollout，主要用于本地快速 smoke test。"""

    # 根据 render 参数选择是否开启 human 模式渲染。
    render_mode = "human" if render else None

    # 创建环境实例。
    env = RelativisticFoodEnv(render_mode=render_mode)

    try:
        # 用给定随机种子做一次 reset。
        obs, info = env.reset(seed=seed)

        # 统计整段 rollout 的累计 reward。
        episode_reward = 0.0

        # 最多执行指定步数。
        for _ in range(steps):
            # 从 action_space 随机采样一个动作。
            action = env.action_space.sample()

            # 推进一步。
            obs, reward, terminated, truncated, info = env.step(action)

            # 累加 reward。
            episode_reward += reward

            # 如果 episode 结束，就提前退出循环。
            if terminated or truncated:
                break

        # rollout 结束后打印一份简要摘要。
        print("Random rollout summary")
        print(f"  reward: {episode_reward:.2f}")
        print(f"  score:  {info['score']}")
        print(f"  obs:    {obs}")

    finally:
        # 无论是否异常，都确保环境被正确关闭。
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    # 创建解析器，并给出脚本说明文字。
    parser = argparse.ArgumentParser(description="Smoke-test the relativistic RL environment.")

    # 允许用户指定随机 rollout 的步数。
    parser.add_argument("--steps", type=int, default=600, help="Number of random-policy steps.")

    # 允许用户指定 reset 的随机种子。
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reset().")

    # 允许用户关闭渲染，只做无界面 smoke test。
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable pygame rendering during the random-policy rollout.",
    )

    # 返回配置好的参数解析器。
    return parser


def main() -> None:
    """脚本入口。"""

    # 先解析命令行参数。
    args = build_arg_parser().parse_args()

    # 再按解析结果执行一次随机 rollout。
    random_rollout(steps=args.steps, render=not args.no_render, seed=args.seed)


# 只有当这个文件被直接执行时，才运行 main()；被 import 时不会自动跑。
if __name__ == "__main__":
    main()
