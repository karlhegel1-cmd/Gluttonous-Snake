import math
import random
from collections import deque
from dataclasses import dataclass, field

import pygame


# 给 pygame 的二维向量类型起一个简短别名，后面写起来更方便
Vec2 = pygame.math.Vector2

# 窗口宽高
WIDTH, HEIGHT = 1280, 800

# 屏幕中心；玩家会始终画在这个位置
CENTER = Vec2(WIDTH * 0.5, HEIGHT * 0.5)

# 目标帧率
FPS = 60

# 单帧最大时间步长，防止卡顿时 dt 过大导致数值更新不稳定
DT_CAP = 1.0 / 60.0

# 程序内部使用的“光速”尺度，不是现实 SI 单位，而是游戏世界单位
C = 420.0

# 最多保留多少秒的世界线历史
MAX_HISTORY_SECONDS = 28.0

# 世界线历史最多记录多少个事件点
# 大约等于 历史秒数 × 帧率，再稍微留一点余量
WORLDLINE_LIMIT = int(MAX_HISTORY_SECONDS * FPS) + 16

# 基础视觉缩放；世界坐标映射到屏幕时会乘它
BASE_VISUAL_SCALE = 2.2

# 玩家基础加速度标尺
PLAYER_ACCEL = 300

# 玩家和目标点的基础半径
BASE_PLAYER_RADIUS = 18.0
BASE_POINT_RADIUS = 16.0

# 目标点的基础漂移速度
POINT_SPEED = 0

# 如果点离玩家太远，就重生
ARENA_RADIUS = 16000.0

# 目标点出生时，希望它落在“屏幕中心附近一圈环带”内
# 这里先用屏幕尺度定义，再在生成时换算回世界坐标尺度
SPAWN_SCREEN_RADIUS_MIN = min(WIDTH, HEIGHT) * 0.34
SPAWN_SCREEN_RADIUS_MAX = min(WIDTH, HEIGHT) * 0.46

# 一些颜色定义
BG = (7, 9, 16)
HUD = (214, 221, 236)
PLAYER_COLOR = (118, 255, 210)
POINT_COLOR = (255, 214, 120)


@dataclass
class Event:
    # 一个时空事件：
    # t   : 实验室系坐标时间
    # x,y : 实验室系空间坐标
    # tau : 该物体自身累计的固有时
    t: float
    x: float
    y: float
    tau: float

    def vec(self) -> Vec2:
        # 把事件的空间坐标转成二维向量，便于做向量运算
        return Vec2(self.x, self.y)


@dataclass
class State:
    # 物体当前动力学状态
    # pos : 当前位置
    # vel : 当前“固有速度 proper velocity”，不是普通坐标速度
    # tau : 当前累计固有时
    pos: Vec2
    vel: Vec2
    tau: float


@dataclass
class WorldlineBody:
    # 具有世界线的对象，比如玩家或目标点
    name: str
    color: tuple[int, int, int]
    radius: float

    # history 存这个物体过去的一串 Event，组成世界线离散采样
    # deque 限制最大长度，旧事件会自动丢弃
    history: deque = field(default_factory=lambda: deque(maxlen=WORLDLINE_LIMIT))

    # 当前状态；初始化前可能为空
    state: State | None = None

    def latest_event(self) -> Event:
        # 返回最新记录的那个事件
        return self.history[-1]

    def record(self, t: float) -> None:
        # 把当前 state 记录成一个新的 Event，压入世界线历史
        if self.state is None:
            return
        self.history.append(Event(t, self.state.pos.x, self.state.pos.y, self.state.tau))


def lerp_event(a: Event, b: Event, s: float) -> Event:
    # 在线性插值参数 s 下，在事件 a 和 b 之间取一个中间事件
    # s=0 返回 a，s=1 返回 b，中间值返回中间事件
    return Event(
        a.t + (b.t - a.t) * s,
        a.x + (b.x - a.x) * s,
        a.y + (b.y - a.y) * s,
        a.tau + (b.tau - a.tau) * s,
    )


def lightcone_function(event: Event, observer: Event) -> float:
    # 计算 event 相对于 observer 的“光锥函数”
    #
    # 形式是：
    #   dx^2 + dy^2 - (c dt)^2
    #
    # 若结果：
    #   = 0 : 在 observer 的光锥上
    #   > 0 : 类空
    #   < 0 : 类时
    #
    # 这里用 dt = observer.t - event.t，
    # 所以若 event 是 observer 的过去可见事件，理论上应满足 f = 0
    dx = event.x - observer.x
    dy = event.y - observer.y
    dt = observer.t - event.t
    return dx * dx + dy * dy - (C * dt) * (C * dt)


def logistic_progress(score: float, center: float, width: float) -> float:
    # 逻辑斯蒂函数（S 形曲线）
    # 用来把分数平滑映射到 0~1，避免游戏参数随分数突变
    return 1.0 / (1.0 + math.exp(-(score - center) / width))


def score_phase_mix(score: float) -> tuple[float, float, float]:
    # 把分数映射成三个阶段权重：
    # mid   : 中期
    # high  : 后期
    # surge : 冲刺期
    # 这些值都在 0~1 之间，用于平滑调节游戏节奏
    mid = logistic_progress(score, 10.0, 3.4)
    high = logistic_progress(score, 20.0, 4.2)
    surge = logistic_progress(score, 34.0, 4.8)
    return mid, high, surge


def score_scaled_body_scale(score: int) -> float:
    # 分数升高后，玩家和目标点会逐渐缩小
    # 让后期更难打中，也让画面更有“速度起来了”的感觉
    shrink_mid = logistic_progress(score, 16.0, 3.0)
    shrink_high = logistic_progress(score, 28.0, 4.2)
    return 1.0 - 0.22 * shrink_mid - 0.3 * shrink_high


def score_scaled_player_accel(score: int) -> float:
    # 分数升高后，玩家推力增强
    # 这样高分阶段更容易进入高速相对论区
    mid, high, surge = score_phase_mix(score)
    return PLAYER_ACCEL * (0.42 + 0.50 * mid + 0.82 * high + 2.15 * surge)


def score_scaled_view_scale(score: int) -> float:
    # 分数升高后，视野缩放降低
    # 表现上是“单位世界长度在屏幕上变小”，像是世界更大、节奏更快
    mid, high, surge = score_phase_mix(score)
    return BASE_VISUAL_SCALE / (1.0 + 0.28 * mid + 0.72 * high + 1.28 * surge)


def score_scaled_point_speed(score: int) -> float:
    # 分数升高后，目标点漂移速度增加
    mid, high, surge = score_phase_mix(score)
    return POINT_SPEED * (0.04 + 0.16 * mid + 0.42 * high + 0.95 * surge)


def score_scaled_spawn_radius(score: int, view_scale: float) -> tuple[float, float]:
    # 目标点的出生距离希望在“屏幕半径意义下”较稳定
    # 这里先算屏幕尺度下的出生环带，再除以 view_scale 转回世界坐标尺度
    mid, high, surge = score_phase_mix(score)
    screen_min = SPAWN_SCREEN_RADIUS_MIN * (1.0 + 0.02 * mid + 0.05 * high + 0.08 * surge)
    screen_max = SPAWN_SCREEN_RADIUS_MAX * (1.0 + 0.03 * mid + 0.07 * high + 0.12 * surge)
    return screen_min / view_scale, screen_max / view_scale


def gamma_from_proper_velocity(proper_velocity: Vec2) -> float:
    # 由固有速度 u 计算 gamma
    #
    # 相对论中有：
    #   u = gamma * v
    # 且
    #   gamma = sqrt(1 + u^2/c^2)
    return math.sqrt(1.0 + proper_velocity.length_squared() / (C * C))


def coordinate_velocity_from_proper_velocity(proper_velocity: Vec2) -> Vec2:
    # 由固有速度 u 还原普通坐标速度 v
    # 因为：
    #   u = gamma * v
    # 所以：
    #   v = u / gamma
    gamma = gamma_from_proper_velocity(proper_velocity)
    if gamma == 0.0:
        return Vec2(0.0, 0.0)
    return proper_velocity / gamma


def proper_velocity_from_coordinate_velocity(velocity: Vec2) -> Vec2:
    # 由普通坐标速度 v 转成固有速度 u
    #
    # 先算：
    #   beta^2 = v^2 / c^2
    #   gamma = 1 / sqrt(1 - beta^2)
    # 再返回：
    #   u = gamma * v
    speed_sq = velocity.length_squared()
    if speed_sq == 0.0:
        return Vec2(0.0, 0.0)
    beta_sq = speed_sq / (C * C)
    gamma = 1.0 / math.sqrt(max(1e-12, 1.0 - beta_sq))
    return velocity * gamma


def transform_event_to_player_frame(event: Event, observer: Event, observer_velocity: Vec2) -> tuple[float, Vec2]:
    # 把实验室系中的某个事件 event
    # 变换到“玩家当前瞬时惯性系”中
    #
    # observer 是玩家当前事件，相当于新参考系原点
    # observer_velocity 是玩家当前实验室系速度
    #
    # 返回：
    #   t_prime   : 事件在玩家系中的时间坐标
    #   pos_prime : 事件在玩家系中的空间坐标（二维向量）

    # 先写出事件相对玩家当前事件的时空位移
    dx = event.x - observer.x
    dy = event.y - observer.y
    dt = event.t - observer.t
    delta = Vec2(dx, dy)

    # 玩家当前速度大小
    speed = observer_velocity.length()

    # 如果玩家几乎静止，则玩家系与实验室系几乎重合
    # 直接返回普通差分即可，避免除零和方向不稳定
    if speed < 1e-8:
        return dt, delta

    # 速度方向单位向量
    direction = observer_velocity / speed

    # 把空间位移分解成：
    # 1. 平行于玩家速度方向的分量
    # 2. 垂直于玩家速度方向的分量
    parallel = delta.dot(direction)
    perpendicular = delta - direction * parallel

    # beta^2 = v^2/c^2，并做数值保护，防止浮点误差导致 >=1
    beta_sq = min(0.999999999999, observer_velocity.length_squared() / (C * C))
    gamma = 1.0 / math.sqrt(1.0 - beta_sq)

    # Lorentz 变换：
    # t' = gamma * (dt - v * x_parallel / c^2)
    t_prime = gamma * (dt - speed * parallel / (C * C))

    # 平行方向坐标变换：
    # x'_parallel = gamma * (x_parallel - v * dt)
    x_parallel_prime = gamma * (parallel - speed * dt)

    # 垂直方向不变，再把平行和垂直部分拼回二维向量
    pos_prime = direction * x_parallel_prime + perpendicular
    return t_prime, pos_prime


def intersect_past_lightcone(worldline: deque, observer: Event) -> Event | None:
    # 在 worldline 上寻找一个事件，使其落在 observer 的过去光锥上
    #
    # 直观上：找到“点过去什么时候发出的光，正好在玩家当前时刻到达玩家”
    #
    # 返回：
    #   找到则返回该事件
    #   找不到返回 None

    if len(worldline) < 2:
        # 至少要有两个事件点，才能形成一段世界线线段供求交
        return None

    events = list(worldline)

    # 最新点对 observer 的光锥函数值
    f_newer = lightcone_function(events[-1], observer)

    # 从新到旧扫描相邻线段
    for idx in range(len(events) - 2, -1, -1):
        older = events[idx]
        newer = events[idx + 1]
        f_older = lightcone_function(older, observer)

        # 如果老点刚好就在光锥上，直接返回
        if abs(f_older) < 1e-5:
            return older

        # 检查这一小段是否跨过 f=0
        # 若跨过，则说明这一段世界线与过去光锥有交点
        crossed = (f_newer <= 0.0 <= f_older) or (f_newer >= 0.0 >= f_older)
        if crossed:
            # 用二分法在 older 和 newer 之间逼近交点
            lo = 0.0
            hi = 1.0
            left = older
            right = newer
            left_value = f_older

            for _ in range(24):
                mid = 0.5 * (lo + hi)
                probe = lerp_event(left, right, mid)
                f_mid = lightcone_function(probe, observer)

                # 若已经足够接近 f=0，直接返回
                if abs(f_mid) < 1e-5:
                    return probe

                # 根据符号变化继续保留左半或右半区间
                if (left_value <= 0.0 <= f_mid) or (left_value >= 0.0 >= f_mid):
                    hi = mid
                    right = probe
                else:
                    lo = mid
                    left = probe
                    left_value = f_mid

            # 二分结束后返回区间中点作为近似交点
            return lerp_event(older, newer, 0.5 * (lo + hi))

        # 继续向更旧的线段扫描
        f_newer = f_older

    return None


def screen_from_relative(relative: Vec2, view_scale: float) -> Vec2:
    # 把“以玩家为原点的相对位置”映射成屏幕坐标
    #
    # x 方向直接对应屏幕 x
    # y 方向要取负，因为数学坐标 y 向上，而屏幕像素 y 向下
    return CENTER + Vec2(relative.x, -relative.y) * view_scale


def edge_intersection(direction: Vec2, margin: float = 34.0) -> Vec2:
    # 给定从屏幕中心出发的方向，求这条射线与屏幕边框（留 margin 边距）的交点
    # 这样当目标在屏幕外时，可以在边缘画一个箭头指示方向

    if direction.length_squared() == 0.0:
        return Vec2(CENTER)

    bounds = pygame.Rect(margin, margin, WIDTH - 2 * margin, HEIGHT - 2 * margin)
    hits = []

    # 试探与左右边界的交点
    if direction.x != 0.0:
        for wall_x in (bounds.left, bounds.right):
            scale = (wall_x - CENTER.x) / direction.x
            if scale > 0.0:
                y = CENTER.y + direction.y * scale
                if bounds.top <= y <= bounds.bottom:
                    hits.append(Vec2(wall_x, y))

    # 试探与上下边界的交点
    if direction.y != 0.0:
        for wall_y in (bounds.top, bounds.bottom):
            scale = (wall_y - CENTER.y) / direction.y
            if scale > 0.0:
                x = CENTER.x + direction.x * scale
                if bounds.left <= x <= bounds.right:
                    hits.append(Vec2(x, wall_y))

    if not hits:
        return Vec2(CENTER)

    # 取离中心最近的那个合法交点
    return min(hits, key=lambda point: (point - CENTER).length_squared())


def draw_triangle(surface: pygame.Surface, tip: Vec2, direction: Vec2, color: tuple[int, int, int]) -> None:
    # 画一个朝向 direction 的三角形箭头，尖端在 tip
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
    # 用多层半透明圆叠加出一个简单发光效果
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
    # 玩家始终画在屏幕中心，带一点轻微脉动效果
    radius = player_radius + 1.5 * pulse
    pygame.draw.circle(surface, PLAYER_COLOR, (int(CENTER.x), int(CENTER.y)), int(radius), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x - 18, CENTER.y), (CENTER.x - 8, CENTER.y), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x + 8, CENTER.y), (CENTER.x + 18, CENTER.y), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x, CENTER.y - 18), (CENTER.x, CENTER.y - 8), 1)
    pygame.draw.line(surface, PLAYER_COLOR, (CENTER.x, CENTER.y + 8), (CENTER.x, CENTER.y + 18), 1)


def make_player() -> WorldlineBody:
    # 创建玩家：
    # 初始位置在原点，初始速度为 0，固有时为 0
    return WorldlineBody(
        name="player",
        color=PLAYER_COLOR,
        radius=BASE_PLAYER_RADIUS,
        state=State(Vec2(0.0, 0.0), Vec2(0.0, 0.0), 0.0),
    )


def make_point(origin: Vec2, score: int, view_scale: float) -> WorldlineBody:
    # 创建目标点
    #
    # 逻辑：
    # 1. 围绕 origin 在一圈环带上随机生成位置
    # 2. 随机给一个漂移方向和速度
    # 3. 把漂移的普通速度转成固有速度存进 state

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
    # 读取按键输入，返回玩家当前“加速方向”的单位向量
    # 支持 WASD 和方向键
    keys = pygame.key.get_pressed()
    accel = Vec2(
        float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT]),
        float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN]),
    )

    # 若有输入则归一化，避免斜向加速比单方向更强
    return accel.normalize() if accel.length_squared() > 0.0 else accel


def advance_body(body: WorldlineBody, start_t: float, end_t: float, accel: Vec2 | None = None) -> None:
    # 把一个物体从 start_t 推进到 end_t
    #
    # 这里 state.vel 存的是固有速度 proper velocity
    # 更新逻辑大致是：
    # 1. 用当前固有速度算 gamma
    # 2. 由 dt 算出该物体自己的 dτ
    # 3. 用加速度推进固有速度
    # 4. 把新旧固有速度都转成普通速度
    # 5. 用平均速度更新位置
    # 6. 记录新的世界线事件

    if body.state is None or end_t <= start_t:
        return

    # 坐标时间步长
    dt = end_t - start_t

    # 当前固有速度
    proper_velocity = Vec2(body.state.vel)

    # 对应的 gamma
    gamma = gamma_from_proper_velocity(proper_velocity)

    # 固有时间步长 dτ = dt / gamma
    dtau = dt / gamma

    # 若没有给 accel，就视为零加速度
    applied_accel = accel if accel is not None else Vec2(0.0, 0.0)

    # 用固有时推进固有速度
    next_proper_velocity = proper_velocity + applied_accel * dtau

    # 新旧固有速度对应的普通坐标速度
    next_coordinate_velocity = coordinate_velocity_from_proper_velocity(next_proper_velocity)
    current_coordinate_velocity = coordinate_velocity_from_proper_velocity(proper_velocity)

    # 用梯形法（平均速度）更新位置
    new_pos = body.state.pos + 0.5 * (current_coordinate_velocity + next_coordinate_velocity) * dt

    # 累加固有时
    new_tau = body.state.tau + dtau

    # 写回状态
    body.state = State(new_pos, next_proper_velocity, new_tau)

    # 在 end_t 时刻记录一个新的世界线事件
    body.record(end_t)


def render_visible_point(
    surface: pygame.Surface,
    point: WorldlineBody,
    observer: Event,
    observer_velocity: Vec2,
    view_scale: float,
    point_radius: float,
    player_frame_mode: bool,
) -> float | None:
    # 渲染“玩家当前真正能看到的点”
    #
    # 核心逻辑：
    # 1. 先在实验室系中找点世界线与玩家当前事件过去光锥的交点
    # 2. 这才是真正“此刻看到”的事件
    # 3. 再决定用哪种方式显示：
    #    - lab delta：直接用实验室系坐标差
    #    - player frame：变换到玩家当前瞬时惯性系
    #
    # 返回 t_prime，供 HUD 调试显示；
    # 若找不到可见事件则返回 None

    # 必须先在实验室系求过去光锥交点，因为历史世界线就是按实验室系存储的
    visible_event = intersect_past_lightcone(point.history, observer)
    if visible_event is None:
        return None

    # 求出真正可见事件之后，再决定如何把它映射成显示坐标
    if player_frame_mode:
        # 玩家参考系模式：把该事件变换到玩家当前瞬时惯性系
        t_prime, relative = transform_event_to_player_frame(visible_event, observer, observer_velocity)
    else:
        # 旧模式：直接使用实验室系中的相对坐标差
        relative = visible_event.vec() - observer.vec()
        t_prime = visible_event.t - observer.t

    screen_pos = screen_from_relative(relative, view_scale)

    # 若在屏幕内，画发光圆点
    if 0.0 <= screen_pos.x <= WIDTH and 0.0 <= screen_pos.y <= HEIGHT:
        draw_glow(surface, screen_pos, point.color, point_radius)
        pygame.draw.circle(
            surface,
            point.color,
            (int(screen_pos.x), int(screen_pos.y)),
            int(point_radius),
        )
    else:
        # 若在屏幕外，则在边缘画一个朝向它的箭头
        tip = edge_intersection(screen_pos - CENTER)
        draw_triangle(surface, tip, tip - CENTER, point.color)

    return t_prime


def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    score: int,
    accel_scale: float,
    gamma: float,
    beta: float,
    t_prime: float | None,
    player_frame_mode: bool,
) -> None:
    # 绘制左上角 / 左下角的 HUD 信息：
    # - score
    # - thrust 倍率
    # - gamma
    # - v/c
    # - 当前可见事件在玩家系中的 t'
    # - 当前显示模式
    score_text = font.render(f"score {score}", True, HUD)
    accel_text = font.render(f"thrust {accel_scale:0.2f}x", True, HUD)
    gamma_text = font.render(f"gamma {gamma:0.3f}", True, HUD)
    beta_text = font.render(f"v/c {beta:0.3f}", True, HUD)

    # 若没有可见事件，就显示 --
    tprime_value = "--" if t_prime is None else f"{t_prime:0.3f}"
    tprime_text = font.render(f"t' {tprime_value}", True, HUD)

    # 当前模式提示
    mode_text = small_font.render(
        "TAB: player frame" if player_frame_mode else "TAB: lab delta",
        True,
        HUD,
    )

    hint_text = small_font.render("WASD / arrows accelerate", True, HUD)

    surface.blit(score_text, (18, 16))
    surface.blit(accel_text, (18, 46))
    surface.blit(gamma_text, (18, 76))
    surface.blit(beta_text, (18, 106))
    surface.blit(tprime_text, (18, 136))
    surface.blit(mode_text, (18, HEIGHT - 62))
    surface.blit(hint_text, (18, HEIGHT - 34))


def main() -> None:
    # pygame 初始化
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Relativistic Glow Eater")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)
    small_font = pygame.font.SysFont("consolas", 18)

    # 这里变量名叫 proper_time，但它在主循环里其实扮演“全局推进时间”的角色
    # 每帧都直接加 dt，所以更像坐标时间计数器
    proper_time = 0.0

    # 分数
    score = 0

    # 是否启用“玩家参考系显示模式”
    player_frame_mode = True

    # 创建玩家和第一个目标点
    player = make_player()
    point = make_point(player.state.pos, score, score_scaled_view_scale(score))

    # 记录初始世界线事件
    player.record(0.0)
    point.record(0.0)

    running = True
    while running:
        # 这一帧的 dt，限制最大值保证数值稳定
        dt = min(clock.tick(FPS) / 1000.0, DT_CAP)
        frame_start = proper_time
        frame_end = proper_time + dt

        # 处理输入事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                # TAB 切换“玩家参考系模式 / 实验室系坐标差模式”
                player_frame_mode = not player_frame_mode

        # 读取当前输入方向
        accel_input = player_acceleration()

        # 按当前分数计算一系列动态参数
        view_scale = score_scaled_view_scale(score)
        thrust = score_scaled_player_accel(score)
        accel_scale = thrust / PLAYER_ACCEL
        body_scale = score_scaled_body_scale(score)
        player_radius = BASE_PLAYER_RADIUS * body_scale
        point_radius = BASE_POINT_RADIUS * body_scale

        # 判定吃到目标的距离阈值
        hit_radius = player_radius + point_radius

        # 推进玩家和目标点
        # 玩家有受控加速度；点没有额外加速度，默认做匀速漂移
        advance_body(player, frame_start, frame_end, accel_input * thrust)
        advance_body(point, frame_start, frame_end)
        proper_time = frame_end

        # 玩家当前事件 = 观察者事件
        observer = player.latest_event()

        # 玩家当前普通坐标速度
        observer_velocity = coordinate_velocity_from_proper_velocity(player.state.vel)

        # 当前 gamma 与 beta=v/c，用于 HUD 调试显示
        gamma = gamma_from_proper_velocity(player.state.vel)
        beta = observer_velocity.length() / C

        # 这里的吃点 / 飞出场地判定，用的是“点当前真实位置”和“玩家当前真实位置”的距离
        # 不是可见位置
        point_distance = point.state.pos.distance_to(observer.vec())
        if point_distance <= hit_radius or point_distance > ARENA_RADIUS:
            if point_distance <= hit_radius:
                score += 1

            # 按新分数重生一个目标点
            next_view_scale = score_scaled_view_scale(score)
            point = make_point(observer.vec(), score, next_view_scale)

            # 在当前 observer.t 时刻记录新点的初始事件
            point.record(observer.t)

        # 清屏
        screen.fill(BG)

        # 画“当前真正可见的点”，并拿到该可见事件在玩家系下的 t'
        t_prime = render_visible_point(
            screen,
            point,
            observer,
            observer_velocity,
            view_scale,
            point_radius,
            player_frame_mode,
        )

        # 画玩家本体（玩家始终居中）
        draw_player(screen, player_radius, 0.5 + 0.5 * math.sin(proper_time * 2.4))

        # 画 HUD
        draw_hud(screen, font, small_font, score, accel_scale, gamma, beta, t_prime, player_frame_mode)

        # 刷新显示
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()