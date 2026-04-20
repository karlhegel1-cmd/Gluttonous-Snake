from __future__ import annotations

"""
这个文件负责评估已经训练好的模型。

评估和训练的区别是：
- 训练：模型一边和环境交互，一边更新参数
- 评估：模型只做决策，不更新参数，只看它表现如何

常用参数里最重要的是：
- model-path：加载哪个模型文件
- episodes：评估多少局
- render：是否打开窗口观察行为
- stochastic：是否使用带随机性的动作输出
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import SAC
except ModuleNotFoundError as exc:
    if exc.name == "stable_baselines3":
        raise ModuleNotFoundError(
            "Missing dependency 'stable_baselines3'. Install it in the current interpreter with:\n"
            f"  {sys.executable} -m pip install stable-baselines3 gymnasium"
        ) from exc
    raise

from demo_to_rl_env import RelativisticFoodEnv


# 评估脚本：
# 加载已经训练好的 SAC 模型，在同一个环境定义上跑若干回合，
# 输出平均奖励、平均得分和平均步数。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "sac_relativistic_food",
        help="模型路径。可以传不带 .zip 的基路径，也可以直接传 .zip 文件。",
    )
    parser.add_argument("--episodes", type=int, default=10, help="评估多少个 episode。")
    parser.add_argument("--max-steps", type=int, default=5000, help="每个 episode 最多走多少步。")
    parser.add_argument("--seed", type=int, default=123, help="随机种子基值。")
    parser.add_argument(
        "--render",
        action="store_true",
        help="打开 pygame 窗口查看模型行为。",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机策略采样动作，而不是确定性动作。",
    )
    parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用辅助奖励。默认开启，可用 --no-reward-shaping 关闭。",
    )
    return parser


def evaluate(args: argparse.Namespace) -> None:
    # 是否可视化完全由命令行决定；渲染时使用 pygame 窗口。
    render_mode = "human" if args.render else None
    env = RelativisticFoodEnv(
        render_mode=render_mode,
        max_steps=args.max_steps,
        reward_shaping=args.reward_shaping,
    )
    model_path = args.model_path
    # SB3 的 load() 可以接收不带 .zip 的基路径，因此这里统一去掉后缀，
    # 兼容用户传入 models/foo 或 models/foo.zip 两种形式。
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    model = SAC.load(str(model_path))

    rewards: list[float] = []
    scores: list[int] = []
    lengths: list[int] = []

    try:
        for episode_idx in range(args.episodes):
            # 每个 episode 使用不同种子，避免多回合完全重复。
            obs, info = env.reset(seed=args.seed + episode_idx)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0

            while not (terminated or truncated):
                # deterministic=True 表示使用确定性策略输出，适合做稳定评估。
                action, _states = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            # 每一局结束后都打印一次结果，便于观察不同回合是否稳定。
            rewards.append(episode_reward)
            scores.append(int(info["score"]))
            lengths.append(episode_length)
            print(
                f"episode={episode_idx:02d} reward={episode_reward:.2f} "
                f"score={info['score']} length={episode_length}"
            )
    finally:
        env.close()

    print("Evaluation summary")
    # 最后输出平均值，帮助你快速判断模型整体水平，而不是只看单局运气。
    print(f"  avg_reward: {np.mean(rewards):.3f}")
    print(f"  avg_score:  {np.mean(scores):.3f}")
    print(f"  avg_len:    {np.mean(lengths):.1f}")


def main() -> None:
    # 允许这个文件被单独执行，而不必经过统一入口脚本。
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
