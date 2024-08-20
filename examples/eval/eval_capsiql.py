from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import dsrl
import numpy as np
import pyrallis
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from pyrallis import field

from osrl.algorithms import CapsIQL, CapsIQLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "log/.../checkpoint/model.pt"
    noise_scale: List[float] = None
    cost_limit: int = 20
    eval_episodes: int = 20
    best: bool = False
    device: str = "cpu"
    threads: int = 4


@pyrallis.wrap()
def eval(args: EvalConfig):

    cfg, model_ckpt = load_config_and_model(args.path, args.best)
    seed_all(cfg["seed"])
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    if "Metadrive" in cfg["task"]:
        import gym
    else:
        import gymnasium as gym  # noqa

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    env.set_target_cost(args.cost_limit)

    model = CapsIQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        num_heads=cfg["num_heads"],
        iql_deterministic=cfg["iql_deterministic"],
        hidden_dim=cfg["hidden_dim"],
        iql_tau=cfg["iql_tau"],
        iql_tau_cost=cfg["iql_tau_cost"],
        beta=cfg["beta"],
        beta_cost=cfg["beta_cost"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        episode_len=cfg["episode_len"],
        device=args.device,
    )
    model.load_state_dict(model_ckpt["model_state"])
    model.to(args.device)

    trainer = CapsIQLTrainer(model,
                         env,
                         reward_scale=cfg["reward_scale"],
                         cost_scale=cfg["cost_scale"],
                         device=args.device)

    agent_perf = trainer.evaluate_switch(args.cost_limit, args.eval_episodes)
    # reward returns, cost returns, episode length
    reward_ret, reward_cost, reward_length = agent_perf
    reward_normalized_ret, reward_normalized_cost = env.get_normalized_score(reward_ret, reward_cost)
    print(
        f"Eval reward: {reward_ret}, normalized reward: {reward_normalized_ret}; \
        cost: {reward_cost}, normalized cost: {reward_normalized_cost}; length: {reward_length}"
    )
    

if __name__ == "__main__":
    eval()

# example
# python eval/eval_capsiql.py --path logs/OfflineAntCircle-v0/CAPSIQL_seed10-d757/CAPSIQL_seed10-d757 --eval_episodes 20