from __future__ import annotations

"""
这个文件负责训练 SAC 强化学习模型。



1. total_timesteps
   总训练步数，不是 episode 数。
   例如 10000 表示环境总共 step 10000 次。

2. max_steps
   每个 episode 最多走多少步。
   如果设为 5000，那么一局最多 5000 步。

3. eval_freq
   每训练多少步，做一次评估。
   比如 10000，表示训练到 10000、20000、30000... 时各评估一次。

4. eval_episodes
   每次评估时，连续跑多少个 episode 再算平均成绩。

5. reward_shaping
   是否开启额外辅助奖励。
   当前默认开启；如果想回到纯稀疏奖励，可以显式传 --no-reward-shaping。
"""

import argparse
import importlib.util
import sys
from pathlib import Path

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
except ModuleNotFoundError as exc:
    if exc.name == "stable_baselines3":
        raise ModuleNotFoundError(
            "Missing dependency 'stable_baselines3'. Install it in the current interpreter with:\n"
            f"  {sys.executable} -m pip install stable-baselines3 gymnasium"
        ) from exc
    raise

from demo_to_rl_env import RelativisticFoodEnv


# 训练脚本：
# 负责创建环境、配置 SAC、执行训练并保存模型与评估日志。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC on the relativistic food environment.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="总训练步数。不是回合数，而是环境 step 的总次数。",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子。用于尽量复现实验结果。")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="每个 episode 的最大步数。达到后会截断当前回合。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "sac_relativistic_food",
        help="模型保存路径前缀。最终通常会生成一个 .zip 模型文件。",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs") / "tensorboard",
        help="TensorBoard 日志目录。训练曲线会保存在这里。",
    )
    parser.add_argument(
        "--eval-log-dir",
        type=Path,
        default=Path("runs") / "eval",
        help="评估日志目录。EvalCallback 会把结果写到这里。",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="每训练多少步评估一次。例如 10000 表示每 10000 步评估一次。",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="每次评估跑多少个 episode，再对成绩取平均。",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="训练前先做一次环境接口检查。",
    )
    parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用辅助奖励。默认开启，可用 --no-reward-shaping 关闭。",
    )
    return parser


def _tensorboard_available() -> bool:
    # 这里只检查模块是否可导入，不主动安装依赖。
    return importlib.util.find_spec("tensorboard") is not None


def _progress_bar_available() -> bool:
    # SB3 的 progress_bar=True 依赖 tqdm 和 rich。
    return importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None


def train(args: argparse.Namespace) -> None:
    # 可选的环境检查流程：在正式训练前验证 Gymnasium / SB3 接口是否合规。
    if args.check_env:
        check_candidate = RelativisticFoodEnv(max_steps=args.max_steps, reward_shaping=args.reward_shaping)
        try:
            check_env(check_candidate, warn=True)
        finally:
            check_candidate.close()

    # 训练前先确保输出目录存在。
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.eval_log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard 不是训练必需依赖；未安装时自动降级为不记录 TensorBoard 日志。
    tensorboard_log: str | None = None
    if _tensorboard_available():
        args.tensorboard_log.mkdir(parents=True, exist_ok=True)
        tensorboard_log = str(args.tensorboard_log)
    else:
        print("TensorBoard is not installed; continuing without TensorBoard logging.")

    # 进度条缺依赖时自动关闭，避免在训练启动阶段报 ImportError。
    use_progress_bar = _progress_bar_available()
    if not use_progress_bar:
        print("Progress-bar dependencies are not installed; continuing without progress bar.")

    # Monitor 会自动记录 episode reward、长度等基础统计量。
    train_env = Monitor(RelativisticFoodEnv(max_steps=args.max_steps, reward_shaping=args.reward_shaping))
    eval_env = Monitor(RelativisticFoodEnv(max_steps=args.max_steps, reward_shaping=args.reward_shaping))

    # 训练过程中定期跑评估，便于观察策略是否真的在变好。
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.eval_log_dir),
        log_path=str(args.eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # 这里使用连续动作算法 SAC，并用一个最小可行的 MLP 结构作为 baseline。
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    # 下面这些是 SAC 的核心超参数。
    #
    # learning_rate:
    #   学习率，控制“每次参数更新迈多大步”。
    #   太大容易训练发散，太小则学得很慢。
    #
    # buffer_size:
    #   经验回放池容量，表示最多缓存多少条历史交互数据。
    #   SAC 会反复从这里采样旧经验来学习。
    #
    # learning_starts:
    #   在收集多少步数据之前，先不开始学习。
    #   因为一开始数据太少，直接训练通常不稳定。
    #
    # batch_size:
    #   每次更新网络时，一次取多少条样本。
    #   可以理解成“每次上课看多少道题一起学”。
    #
    # tau:
    #   目标网络软更新系数。
    #   数值小，更新更平滑；数值大，更新更快。
    #
    # gamma:
    #   折扣因子，表示智能体有多看重未来奖励。
    #   越接近 1，越重视长期回报。
    #
    # train_freq=(1, "step"):
    #   每与环境交互 1 步，就触发一次训练逻辑。
    #
    # gradient_steps=1:
    #   每次触发训练时，做 1 次梯度更新。
    #
    # ent_coef="auto":
    #   熵系数自动学习，帮助 SAC 自动平衡“探索”和“利用”。
    #
    # policy_kwargs={"net_arch": [256, 256]}:
    #   神经网络结构，这里是两层隐藏层，每层 256 个神经元。
    #   对于当前任务，这是一个常见、够用的基线配置。

    try:
        # 有 rich/tqdm 时开启进度条；否则静默降级，不影响训练本身。
        model.learn(total_timesteps=args.total_timesteps, callback=eval_callback, progress_bar=use_progress_bar)
        model.save(str(args.model_path))
        print(f"Saved model to {args.model_path}.zip")
    finally:
        # 即使训练中报错，也确保环境资源被正确关闭。
        train_env.close()
        eval_env.close()


def main() -> None:
    # 允许这个文件单独运行，例如直接 `python train_sac.py ...`。
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
