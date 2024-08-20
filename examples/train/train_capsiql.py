import os
import types
from dataclasses import asdict, dataclass
import sys
parentdir = "/home/../OSRL/" 
sys.path.insert(0, parentdir)

import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from examples.configs.capsiql_configs import CapsIQL_DEFAULT_CONFIG, CapsIQLTrainConfig
from osrl.algorithms import CapsIQL, CapsIQLTrainer
from osrl.common import TransitionDataset
from osrl.common.exp_util import auto_name, seed_all


@pyrallis.wrap()
def train(args: CapsIQLTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(CapsIQLTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(CapsIQL_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(CapsIQL_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task 
    if args.logdir is not None:
        args.logdir += f"_{args.num_heads}"
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym

    env = gym.make(args.task)
    if "Gymnasium" in args.task:
        cost_limits=[20, 40, 80]
    else:
        cost_limits=[10, 20, 40]

    # pre-process offline dataset
    data = env.get_dataset()

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    model = CapsIQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        num_heads=args.num_heads,
        iql_deterministic=args.iql_deterministic,
        hidden_dim=args.hidden_dim,
        iql_tau=args.iql_tau,
        iql_tau_cost=args.iql_tau_cost,
        beta=args.beta,
        beta_cost=args.beta_cost,
        tau=args.tau,
        gamma=args.gamma,
        episode_len=args.episode_len,
        device=args.device,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}
    
    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = CapsIQLTrainer(model,
                         env,
                         logger=logger,
                         actor_lr=args.actor_lr,
                         q_lr=args.q_lr,
                         value_lr=args.value_lr,
                         max_steps=args.max_steps,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         device=args.device)

    dataset = TransitionDataset(data,
                                reward_scale=args.reward_scale,
                                cost_scale=args.cost_scale)
    
    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        observations, next_observations, actions, rewards, costs, done = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(observations, next_observations, actions, rewards, costs,
                               done, step)

        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:            
            for c_lim in cost_limits:
                agent_perf = trainer.evaluate_switch(c_lim, args.eval_episodes)
                agent_ret, agent_cost, agent_length = agent_perf
                logger.store(tab=f"switch_{c_lim}", Cost=agent_cost, Reward=agent_ret, Length=agent_length)

            # save the current weight
            logger.save_checkpoint()

            logger.write(step, display=False)
        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
    #python train/train_capsiql.py --task OfflineHalfCheetahVelocityGymnasium-v1 --beta_cost 5 --iql_tau 0.7 --num_heads 5 --seed 20