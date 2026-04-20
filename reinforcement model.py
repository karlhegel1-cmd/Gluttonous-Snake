from __future__ import annotations

import argparse


# 统一入口文件：
# 这个脚本不承载环境本身、训练算法或评估逻辑，
# 只负责解析命令行参数，并把请求分发到对应模块。
def build_arg_parser() -> argparse.ArgumentParser:
    # This file is only a thin CLI wrapper. The actual environment, training,
    # and evaluation logic live in their dedicated modules.
    parser = argparse.ArgumentParser(description="Unified entry point for the relativistic RL baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training subcommand: forwards common SAC training options to train_sac.py.
    train_parser = subparsers.add_parser("train", help="Train a SAC agent.")
    # 这些参数会原样传给 train_sac.py。
    train_parser.add_argument("--total-timesteps", type=int, default=200_000)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--max-steps", type=int, default=5000)
    train_parser.add_argument("--model-path", default="models/sac_relativistic_food")
    train_parser.add_argument("--tensorboard-log", default="runs/tensorboard")
    train_parser.add_argument("--eval-log-dir", default="runs/eval")
    train_parser.add_argument("--eval-freq", type=int, default=10_000)
    train_parser.add_argument("--eval-episodes", type=int, default=5)
    train_parser.add_argument("--check-env", action="store_true")
    train_parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Evaluation subcommand: loads a trained checkpoint and runs episodes.
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained SAC agent.")
    # 这些参数会原样传给 evaluate.py。
    eval_parser.add_argument("--model-path", default="models/sac_relativistic_food")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--max-steps", type=int, default=5000)
    eval_parser.add_argument("--seed", type=int, default=123)
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.add_argument("--stochastic", action="store_true")
    eval_parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Random rollout is a lightweight smoke test for the environment itself.
    random_parser = subparsers.add_parser("random", help="Run a random-policy smoke test.")
    # random 模式不训练模型，只是随机采样动作，检查环境会不会报错。
    random_parser.add_argument("--steps", type=int, default=600)
    random_parser.add_argument("--seed", type=int, default=0)
    random_parser.add_argument("--no-render", action="store_true")

    return parser


def main() -> None:
    # 所有命令都先从这里进入，然后根据子命令再延迟导入对应模块。
    # 这样做的好处是：只执行 random 时，不会强制要求先安装训练依赖。
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        from pathlib import Path
        from train_sac import train

        # 将字符串路径转成 Path，方便下游脚本做目录创建与文件拼接。
        # train() expects Path objects for output locations.
        args.model_path = Path(args.model_path)
        args.tensorboard_log = Path(args.tensorboard_log)
        args.eval_log_dir = Path(args.eval_log_dir)
        train(args)
        return

    if args.command == "evaluate":
        from pathlib import Path
        from evaluate import evaluate

        # 评估脚本同样统一使用 Path 处理模型文件路径。
        # evaluate() loads the checkpoint from a filesystem path.
        args.model_path = Path(args.model_path)
        evaluate(args)
        return

    if args.command == "random":
        from demo_to_rl_env import random_rollout

        # random 模式只是做环境冒烟测试，动作仍然是从 action_space 采样，
        # 并没有引入任何“环境接管行为”的特殊规则。
        # Keep the environment policy-driven: the "random" mode only samples
        # actions for debugging and does not change the environment rules.
        random_rollout(steps=args.steps, render=not args.no_render, seed=args.seed)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
