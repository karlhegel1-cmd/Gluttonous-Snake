import math
import random
from collections import deque
from dataclasses import dataclass, field

import pygame


Vec2 = pygame.math.Vector2

# 原始二维相对论 demo。
# 这个文件定义了世界线记录、过去光锥求交、相对论速度换算、
# 基础渲染和键盘驱动主循环；RL 环境就是在此基础上改造出来的。

# 画面与时间步长参数。
WIDTH, HEIGHT = 1280, 800  # 窗口宽高，单位是像素。
CENTER = Vec2(WIDTH * 0.5, HEIGHT * 0.5)  # 屏幕中心点，玩家始终画在这里。
FPS = 60  # 目标帧率。数值越大，画面越流畅，但计算也会更频繁。
DT_CAP = 1.0 / 60.0  # 单帧最大 dt，防止掉帧时一次跨太大。

# 物理与游戏参数。
C = 420.0  # 信号传播速度，扮演“光速”的角色。
MAX_HISTORY_SECONDS = 28.0  # 世界线历史最多保留多少秒。
WORLDLINE_LIMIT = int(MAX_HISTORY_SECONDS * FPS) + 16  # 历史事件点最多存多少个。
BASE_VISUAL_SCALE = 2.2  # 世界坐标映射到屏幕时的缩放倍率。
PLAYER_ACCEL = 200  # 玩家基础加速度标尺。
MAX_SPEED = C * 0.82  # 理论最大速度上限（当前原 demo 未直接使用）。
BASE_PLAYER_RADIUS = 18.0  # 玩家基础半径。
BASE_POINT_RADIUS = 16.0  # 食物/目标点基础半径。
POINT_SPEED = 52.0  # 食物基础漂移速度。
ARENA_RADIUS = 1600.0  # 超出这个半径可视为过远或离场。
SPAWN_SCREEN_RADIUS_MIN = min(WIDTH, HEIGHT) * 0.34  # 食物生成时距离中心的最小屏幕半径。
SPAWN_SCREEN_RADIUS_MAX = min(WIDTH, HEIGHT) * 0.46  # 食物生成时距离中心的最大屏幕半径。

BG = (7, 9, 16)
HUD = (214, 221, 236)
PLAYER_COLOR = (118, 255, 210)
POINT_COLOR = (255, 214, 120)


@dataclass
class Event:
    # 世界线上的一个事件点：坐标时 t、位置 (x, y) 和固有时 tau。
    t: float
    x: float
    y: float
    tau: float

    def vec(self) -> Vec2:
        return Vec2(self.x, self.y)


@dataclass
class State:
    # 物体当前状态：位置、速度（这里存的是 proper velocity）和固有时。
    pos: Vec2
    vel: Vec2
    tau: float


@dataclass
class WorldlineBody:
    # 一个可记录世界线的实体，例如玩家或目标点。
    name: str
    color: tuple[int, int, int]
    radius: float
    history: deque = field(default_factory=lambda: deque(maxlen=WORLDLINE_LIMIT))
    state: State | None = None

    def latest_event(self) -> Event:
        return self.history[-1]

    def record(self, t: float) -> None:
        if self.state is None:
            return
        self.history.append(Event(t, self.state.pos.x, self.state.pos.y, self.state.tau))


def lerp_event(a: Event, b: Event, s: float) -> Event:
    # 沿着两事件点做线性插值，用于光锥求交时二分逼近。
    return Event(
        a.t + (b.t - a.t) * s,
        a.x + (b.x - a.x) * s,
        a.y + (b.y - a.y) * s,
        a.tau + (b.tau - a.tau) * s,
    )


def lightcone_function(event: Event, observer: Event) -> float:
    # 闵可夫斯基时空间隔函数：
    # = 0 表示恰好在光锥上，正负号表示在光锥内外。
    dx = event.x - observer.x
    dy = event.y - observer.y
    dt = observer.t - event.t
    return dx * dx + dy * dy - (C * dt) * (C * dt)


def logistic_progress(score: float, center: float, width: float) -> float:
    # 使用平滑 S 曲线控制难度随分数增长，避免突变。
    # 参数说明：
    # - score: 当前分数
    # - center: 过渡大致从哪里开始明显发生
    # - width: 过渡有多“平滑”或“陡峭”
    return 1.0 / (1.0 + math.exp(-(score - center) / width))


def score_phase_mix(score: float) -> tuple[float, float, float]:
    # 将分数映射到三个阶段性强度，用于后续调视角、加速度和目标速度。
    mid = logistic_progress(score, 10.0, 3.4)
    high = logistic_progress(score, 20.0, 4.2)
    surge = logistic_progress(score, 34.0, 4.8)
    return mid, high, surge


def score_scaled_body_scale(score: int) -> float:
    # 随分数升高，玩家和食物在屏幕上的半径会逐渐缩小，提高难度。
    shrink_mid = logistic_progress(score, 16.0, 3.0)
    shrink_high = logistic_progress(score, 28.0, 4.2)
    return 1.0 - 0.22 * shrink_mid - 0.3 * shrink_high


def score_scaled_player_accel(score: int) -> float:
    # 随分数升高，玩家可用最大加速度增加。
    # 这就是“吃到食物后加速能力变强”的来源。
    mid, high, surge = score_phase_mix(score)
    return PLAYER_ACCEL * (0.42 + 0.50 * mid + 0.82 * high + 2.15 * surge)


def score_scaled_view_scale(score: int) -> float:
    # 随分数升高，视角缩放发生变化。
    # 你可以把它理解成“镜头感”在变化。
    mid, high, surge = score_phase_mix(score)
    return BASE_VISUAL_SCALE / (1.0 + 0.28 * mid + 0.72 * high + 1.28 * surge)


def score_scaled_point_speed(score: int) -> float:
    # 随分数升高，食物自己漂移得更快。
    mid, high, surge = score_phase_mix(score)
    return POINT_SPEED * (0.04 + 0.16 * mid + 0.42 * high + 0.95 * surge)


def score_scaled_spawn_radius(score: int, view_scale: float) -> tuple[float, float]:
    # 根据当前分数和视角，计算食物生成半径范围。
    # 返回值是 (最小半径, 最大半径)。
    mid, high, surge = score_phase_mix(score)
    screen_min = SPAWN_SCREEN_RADIUS_MIN * (1.0 + 0.02 * mid + 0.05 * high + 0.08 * surge)
    screen_max = SPAWN_SCREEN_RADIUS_MAX * (1.0 + 0.03 * mid + 0.07 * high + 0.12 * surge)
    return screen_min / view_scale, screen_max / view_scale


def gamma_from_proper_velocity(proper_velocity: Vec2) -> float:
    # proper velocity -> Lorentz gamma。
    return math.sqrt(1.0 + proper_velocity.length_squared() / (C * C))


def coordinate_velocity_from_proper_velocity(proper_velocity: Vec2) -> Vec2:
    # 将 proper velocity 转回通常意义下的坐标速度。
    gamma = gamma_from_proper_velocity(proper_velocity)
    if gamma == 0.0:
        return Vec2(0.0, 0.0)
    return proper_velocity / gamma


def proper_velocity_from_coordinate_velocity(velocity: Vec2) -> Vec2:
    # 将坐标速度转换为 proper velocity，便于相对论更新。
    speed_sq = velocity.length_squared()
    if speed_sq == 0.0:
        return Vec2(0.0, 0.0)
    beta_sq = speed_sq / (C * C)
    gamma = 1.0 / math.sqrt(max(1e-12, 1.0 - beta_sq))
    return velocity * gamma


def intersect_past_lightcone(worldline: deque, observer: Event) -> Event | None:
    # 在给定世界线中寻找与观察者“过去光锥”的交点。
    # 如果找到，说明该事件此刻刚好可以被观察到。
    if len(worldline) < 2:
        return None

    events = list(worldline)
    f_newer = lightcone_function(events[-1], observer)

    for idx in range(len(events) - 2, -1, -1):
        older = events[idx]
        newer = events[idx + 1]
        f_older = lightcone_function(older, observer)

        if abs(f_older) < 1e-5:
            return older

        crossed = (f_newer <= 0.0 <= f_older) or (f_newer >= 0.0 >= f_older)
        if crossed:
            lo = 0.0
            hi = 1.0
            left = older
            right = newer
            left_value = f_older

            # 使用二分逼近提高交点时间和位置估计精度。
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                probe = lerp_event(left, right, mid)
                f_mid = lightcone_function(probe, observer)
                if abs(f_mid) < 1e-5:
                    return probe
                if (left_value <= 0.0 <= f_mid) or (left_value >= 0.0 >= f_mid):
                    hi = mid
                    right = probe
                else:
                    lo = mid
                    left = probe
                    left_value = f_mid

            return lerp_event(older, newer, 0.5 * (lo + hi))

        f_newer = f_older

    return None


def screen_from_relative(relative: Vec2, view_scale: float) -> Vec2:
    # 世界坐标 -> 屏幕坐标；y 轴取反是因为屏幕坐标向下为正。
    return CENTER + Vec2(relative.x, -relative.y) * view_scale


def edge_intersection(direction: Vec2, margin: float = 34.0) -> Vec2:
    # 当目标不在屏幕内时，计算屏幕边缘上的提示箭头位置。
    if direction.length_squared() == 0.0:
        return Vec2(CENTER)

    bounds = pygame.Rect(margin, margin, WIDTH - 2 * margin, HEIGHT - 2 * margin)
    hits = []

    if direction.x != 0.0:
        for wall_x in (bounds.left, bounds.right):
            scale = (wall_x - CENTER.x) / direction.x
            if scale > 0.0:
                y = CENTER.y + direction.y * scale
                if bounds.top <= y <= bounds.bottom:
                    hits.append(Vec2(wall_x, y))

    if direction.y != 0.0:
        for wall_y in (bounds.top, bounds.bottom):
            scale = (wall_y - CENTER.y) / direction.y
            if scale > 0.0:
                x = CENTER.x + direction.x * scale
                if bounds.left <= x <= bounds.right:
                    hits.append(Vec2(x, wall_y))

    if not hits:
        return Vec2(CENTER)

    return min(hits, key=lambda point: (point - CENTER).length_squared())


def draw_triangle(surface: pygame.Surface, tip: Vec2, direction: Vec2, color: tuple[int, int, int]) -> None:
    # 用一个小三角形表示屏幕外目标的大致方向。
    if direction.length_squared() == 0.0:
        return

    forward = direction.normalize()
    side = Vec2(-forward.y, forward.x)
    points = [
        tip,
        tip - forward * 24.0 + side * 10.0,
        tip - forward * 24.0 - side * 10.0,
    ]
    pygame.draw.polygon(surface, color, points)


def draw_glow(surface: pygame.Surface, pos: Vec2, color: tuple[int, int, int], radius: float) -> None:
    # 通过多层透明圆叠加出发光效果。
    for i in range(4, 0, -1):
        scale = 1.0 + i * 0.65
        alpha = max(12, 120 - i * 20)
        glow = pygame.Surface((int(radius * 8), int(radius * 8)), pygame.SRCALPHA)
        center = Vec2(glow.get_width() / 2, glow.get_height() / 2)
        pygame.draw.circle(
            glow,
            (*color, alpha),
            (int(center.x), int(center.y)),
            max(1, int(radius * scale)),
        )
        surface.blit(glow, glow.get_rect(center=(pos.x, pos.y)))


def draw_player(surface: pygame.Surface, player_radius: float, pulse: float) -> None:
    # 玩家始终绘制在屏幕中央，外加轻微脉冲效果。
    radius = player_radius + 1.5 * pulse
    pygame.draw.circle(surface, PLAYER_COLOR, (int(CENTER.x), int(CENTER.y)), int(radius), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x - 18, CENTER.y), (CENTER.x - 8, CENTER.y), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x + 8, CENTER.y), (CENTER.x + 18, CENTER.y), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x, CENTER.y - 18), (CENTER.x, CENTER.y - 8), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x, CENTER.y + 8), (CENTER.x, CENTER.y + 18), 1)


def make_player() -> WorldlineBody:
    # 创建初始玩家实体，位于原点、速度为零。
    return WorldlineBody(
        name="player",
        color=PLAYER_COLOR,
        radius=BASE_PLAYER_RADIUS,
        state=State(Vec2(0.0, 0.0), Vec2(0.0, 0.0), 0.0),
    )


def make_point(origin: Vec2, score: int, view_scale: float) -> WorldlineBody:
    # 根据当前分数和视角，在玩家附近随机生成目标点。
    spawn_min, spawn_max = score_scaled_spawn_radius(score, view_scale)
    angle = random.uniform(0.0, math.tau)
    distance = random.uniform(spawn_min, spawn_max)
    position = origin + Vec2(math.cos(angle), math.sin(angle)) * distance

    drift_angle = random.uniform(0.0, math.tau)
    drift_speed = score_scaled_point_speed(score)
    drift_velocity = Vec2(math.cos(drift_angle), math.sin(drift_angle)) * drift_speed

    return WorldlineBody(
        name="point",
        color=POINT_COLOR,
        radius=BASE_POINT_RADIUS,
        state=State(position, proper_velocity_from_coordinate_velocity(drift_velocity), 0.0),
    )


def player_acceleration() -> Vec2:
    # 读取键盘输入并转成单位加速度方向。
    keys = pygame.key.get_pressed()
    accel = Vec2(
        float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT]),
        float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
    )
    return accel.normalize() if accel.length_squared() > 0.0 else accel


def advance_body(body: WorldlineBody, start_t: float, end_t: float, accel: Vec2 | None = None) -> None:
    # 物理推进核心：
    # 先更新 proper velocity，再转为 coordinate velocity 积分位置。
    if body.state is None or end_t <= start_t:
        return

    dt = end_t - start_t
    proper_velocity = Vec2(body.state.vel)
    gamma = gamma_from_proper_velocity(proper_velocity)
    dtau = dt / gamma
    applied_accel = accel if accel is not None else Vec2(0.0, 0.0)

    next_proper_velocity = proper_velocity + applied_accel * dtau
    next_coordinate_velocity = coordinate_velocity_from_proper_velocity(next_proper_velocity)
    current_coordinate_velocity = coordinate_velocity_from_proper_velocity(proper_velocity)
    new_pos = body.state.pos + 0.5 * (current_coordinate_velocity + next_coordinate_velocity) * dt
    new_tau = body.state.tau + dtau

    body.state = State(new_pos, next_proper_velocity, new_tau)
    body.record(end_t)


def render_visible_point(
    surface: pygame.Surface,
    point: WorldlineBody,
    observer: Event,
    view_scale: float,
    point_radius: float,
) -> None:
    # 只绘制“当前可见”的目标位置，而不是目标真实当前位置。
    visible_event = intersect_past_lightcone(point.history, observer)
    if visible_event is None:
        return

    relative = visible_event.vec() - observer.vec()
    screen_pos = screen_from_relative(relative, view_scale)

    if 0.0 <= screen_pos.x <= WIDTH and 0.0 <= screen_pos.y <= HEIGHT:
        draw_glow(surface, screen_pos, point.color, point_radius)
        pygame.draw.circle(
            surface,
            point.color,
            (int(screen_pos.x), int(screen_pos.y)),
            int(point_radius),
        )
    else:
        tip = edge_intersection(screen_pos - CENTER)
        draw_triangle(surface, tip, tip - CENTER, point.color)


def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    score: int,
    accel_scale: float,
) -> None:
    # 左上角 HUD：显示分数、当前推力倍率和控制提示。
    score_text = font.render(f"score {score}", True, HUD)
    accel_text = font.render(f"thrust {accel_scale:0.2f}x", True, HUD)
    hint_text = small_font.render("WASD / arrows accelerate", True, HUD)
    surface.blit(score_text, (18, 16))
    surface.blit(accel_text, (18, 46))
    surface.blit(hint_text, (18, HEIGHT - 34))


def main() -> None:
    # 原始 demo 主循环：键盘控制玩家，吃到目标后刷新并提升难度。
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Relativistic Glow Eater")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    small_font = pygame.font.Font(None, 18)

    proper_time = 0.0
    score = 0

    player = make_player()
    point = make_point(player.state.pos, score, score_scaled_view_scale(score))

    player.record(0.0)
    point.record(0.0)

    running = True
    while running:
        # 对 dt 做上限保护，避免掉帧时单步积分跨度过大。
        dt = min(clock.tick(FPS) / 1000.0, DT_CAP)
        frame_start = proper_time
        frame_end = proper_time + dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        accel_input = player_acceleration()
        view_scale = score_scaled_view_scale(score)
        thrust = score_scaled_player_accel(score)
        accel_scale = thrust / PLAYER_ACCEL
        body_scale = score_scaled_body_scale(score)
        player_radius = BASE_PLAYER_RADIUS * body_scale
        point_radius = BASE_POINT_RADIUS * body_scale
        hit_radius = player_radius + point_radius

        # 玩家和目标都沿各自世界线前进，只是目标没有受控加速度。
        advance_body(player, frame_start, frame_end, accel_input * thrust)
        advance_body(point, frame_start, frame_end)
        proper_time = frame_end

        observer = player.latest_event()
        point_distance = point.state.pos.distance_to(observer.vec())
        if point_distance <= hit_radius or point_distance > ARENA_RADIUS:
            if point_distance <= hit_radius:
                score += 1
            next_view_scale = score_scaled_view_scale(score)
            point = make_point(observer.vec(), score, next_view_scale)
            point.record(observer.t)

        screen.fill(BG)
        render_visible_point(screen, point, observer, view_scale, point_radius)
        draw_player(screen, player_radius, 0.5 + 0.5 * math.sin(proper_time * 2.4))
        draw_hud(screen, font, small_font, score, accel_scale)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
