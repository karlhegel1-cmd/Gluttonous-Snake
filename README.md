# Relativistic Food RL Baseline

This project turns the original `相对论贪吃蛇.py` demo into a minimal Gymnasium + Stable-Baselines3 continuous-control baseline.

The baseline keeps the original physics ideas:

- worldline history is still recorded with `WorldlineBody`;
- food visibility still comes from the intersection between the food worldline and the agent's past lightcone;
- finite signal propagation speed is still controlled by `C`;
- the agent action is now a policy output that controls acceleration, not velocity or position;
- the maximum acceleration still increases with score through `score_scaled_player_accel()`.

## Files

- `相对论贪吃蛇.py`: original demo and reusable physics helpers.
- `demo_to_rl_env.py`: Gymnasium environment and optional random-policy smoke test.
- `train_sac.py`: SAC training entry point.
- `evaluate.py`: model evaluation entry point.
- `reinforcement model.py`: unified CLI wrapper for train / evaluate / random rollout.
- `utils.py`: action projection and vector helpers.

## Install

Use Python 3.10+.

```bash
py -m pip install numpy pygame gymnasium stable-baselines3 torch
```

Optional extras:

```bash
py -m pip install tensorboard tqdm rich
```

`tensorboard` is only needed if you want TensorBoard logs.
`tqdm` and `rich` are only needed if you want the SB3 training progress bar.
If these packages are missing, `train_sac.py` now falls back automatically and training still runs.

## Observation

The observation is:

```text
obs = [m, x_obs, y_obs]
```

- `m`: visibility mask, `1` when the food is currently observable and `0` otherwise.
- `x_obs, y_obs`: the visible food event's spatial coordinates in the lab frame.
- When the food is not observable, `m = 0`, `x_obs = 0`, and `y_obs = 0`.

This is not the food's true current position, and it is not a unit direction vector.
The environment still uses the past-lightcone intersection to find the visible event on the food worldline.
The observation simply exposes that visible event's lab-frame `x/y` coordinates.

## Action

The action space is continuous:

```text
Box(low=-1, high=1, shape=(2,), dtype=float32)
```

The policy outputs `a_raw in [-1, 1]^2`.

The environment then applies:

1. project `a_raw` into the unit disk if its norm is larger than `1`;
2. map it to the real acceleration limit: `a_t = a_max(score) * a_raw_projected`.

The norm limit is applied to the whole 2D action vector, not component-wise.

## Dynamics

The step update keeps the original relativistic integration skeleton through `advance_body()`:

1. receive the action as an acceleration control input;
2. update proper velocity from the applied acceleration;
3. convert proper velocity to coordinate velocity;
4. update position from velocity over `dt`;
5. update visibility, reward, and respawn logic.

In other words, the policy never outputs the next velocity or next position directly.

## Reward

Default reward is sparse:

- eat food: `+1`
- otherwise: `0`

Optional shaping reward exists for experiments, and it is enabled by default.

## Episode Ending

Episodes are truncated when:

- `max_steps` is reached;
- the agent leaves a large safety radius.

Episodes terminate when the numeric state becomes invalid.

These guards are small RL-oriented additions for training stability.

## Train

```bash
py train_sac.py --check-env --total-timesteps 200000
```

Or use the wrapper:

```bash
py "reinforcement model.py" train --check-env --total-timesteps 200000
```

TensorBoard:

```bash
py -m tensorboard.main --logdir runs/tensorboard
```

If `tensorboard` is not installed, training still works and just skips TensorBoard logging.
If `tqdm` / `rich` are not installed, training still works and just skips the terminal progress bar.

## Evaluate

```bash
py evaluate.py --model-path models/sac_relativistic_food --episodes 10 --render
```

Or:

```bash
py "reinforcement model.py" evaluate --model-path models/sac_relativistic_food --episodes 10 --render
```

## Random Smoke Test

This is useful before training:

```bash
py demo_to_rl_env.py
```

Or:

```bash
py "reinforcement model.py" random
```

## Notes About The Minimal Changes

- The original demo variable named `proper_time` was actually used as the simulation time `t`. In the RL environment it is renamed to `coordinate_time` for clarity.
- Food spawning now uses the Gymnasium RNG so `reset(seed=...)` is reproducible.
- The old keyboard HUD text was removed because the environment is fully policy-driven now.
