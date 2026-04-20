from __future__ import annotations

import numpy as np


# 工具函数文件：
# 这里只放与环境/训练都可能复用的纯数学辅助逻辑，
# 不依赖具体的环境状态，便于单独测试和复用。
def project_to_unit_ball(action: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    """Project a 2D action into the unit disk and guard against invalid values."""
    # 将任意输入统一转成形状为 (2,) 的 float32 向量。
    projected = np.asarray(action, dtype=np.float32).reshape(2)
    # 如果动作里出现 NaN / inf，直接回退为零动作，避免环境数值炸掉。
    if not np.all(np.isfinite(projected)):
        return np.zeros(2, dtype=np.float32)

    # 连续控制动作要求限制在单位圆内，而不是分别裁剪 x/y 分量。
    norm = float(np.linalg.norm(projected))
    if norm > 1.0:
        projected = projected / norm
    return projected.astype(np.float32)


def safe_unit_vector(vector: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float]:
    """Return (unit_vector, norm); use zeros when the norm is too small."""
    # 与 project_to_unit_ball 类似，先做统一格式化和数值保护。
    arr = np.asarray(vector, dtype=np.float32).reshape(2)
    if not np.all(np.isfinite(arr)):
        return np.zeros(2, dtype=np.float32), 0.0

    norm = float(np.linalg.norm(arr))
    # 极小向量没有稳定方向，直接返回零向量，避免除以极小数。
    if norm <= 1e-8:
        return np.zeros(2, dtype=np.float32), 0.0
    return (arr / norm).astype(np.float32), norm
