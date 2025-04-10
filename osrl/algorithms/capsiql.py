import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa
from osrl.common.net import TwinQ, ValueFunction, DeterministicPolicy, MHGaussianPolicy
from typing import Tuple
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

EXP_ADV_MAX = 100.0
EXP_ADVC_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean( torch.abs(tau - (u < 0).float()) * u**2)

class CapsIQL(nn.Module):
    """
    Multi Head IQL Agent
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 num_heads: int = 2,
                 iql_deterministic: bool = False,
                 hidden_dim: int = 512,
                 iql_tau: float = 0.9,
                 iql_tau_cost: float = 0.9,
                 beta: float = 3.0,
                 beta_cost: float = 3.0,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_heads = num_heads
        self.iql_deterministic = iql_deterministic
        self.hidden_dim = hidden_dim
        self.iql_tau = iql_tau
        self.iql_tau_cost = iql_tau_cost
        self.beta = beta
        self.beta_cost = beta_cost
        self.tau = tau
        self.gamma = gamma
        self.episode_len = episode_len
        self.device = device

        self.reward_q = TwinQ(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.reward_q_target = copy.deepcopy(self.reward_q).requires_grad_(False).to(self.device)
        
        self.cost_q = TwinQ(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.cost_q_target = copy.deepcopy(self.cost_q).requires_grad_(False).to(self.device)

        self.reward_v = ValueFunction(self.state_dim, self.hidden_dim).to(self.device)
        self.cost_v = ValueFunction(self.state_dim, self.hidden_dim).to(self.device)
        self.actor = (
                DeterministicPolicy(
                    state_dim, action_dim, max_action, self.hidden_dim
                )
                if self.iql_deterministic
                else MHGaussianPolicy(
                    state_dim, action_dim, self.num_heads, max_action, hidden_dim=self.hidden_dim, device=self.device
                )
                ).to(self.device)
    
    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module 
        towards the parameters of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
    
    
    def value_loss(self, observations, actions):
        with torch.no_grad():
            target_q = self.reward_q_target(observations, actions)

        v = self.reward_v(observations)
        adv = target_q - v
        loss_value = asymmetric_l2_loss(adv, self.iql_tau)

        self.reward_value_optim.zero_grad()
        loss_value.backward()
        self.reward_value_optim.step()
        stats_value = {"loss/reward_value_loss": loss_value.item()}

        return adv, stats_value

    def cost_value_loss(self, observations, actions):
        with torch.no_grad():
            target_qc = self.cost_q_target(observations, actions)

        vc = self.cost_v(observations)
        adv = target_qc - vc
        loss_value = asymmetric_l2_loss(adv, self.iql_tau_cost)

        self.cost_value_optim.zero_grad()
        loss_value.backward()
        self.cost_value_optim.step()
        stats_value = {"loss/cost_value_loss": loss_value.item()}

        return adv, stats_value


    def reward_q_loss(self, observations, actions, rewards, done, next_v):
        targets = rewards.float() + (1.0 - done) * self.gamma * next_v.detach()
        qs = self.reward_q.both(observations, actions)
        loss_critic = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.reward_q_optim.zero_grad()
        loss_critic.backward()
        self.reward_q_optim.step()
        stats_critic = {"loss/reward_q_loss": loss_critic.item()}

        return stats_critic

    def cost_q_loss(self, observations, actions, costs, done, next_v):
        costs = -costs.float()
        targets = costs + (1.0 - done) * self.gamma * next_v.detach()
        qcs = self.cost_q.both(observations, actions)
        loss_critic = sum(F.mse_loss(q, targets) for q in qcs) / len(qcs)
        self.cost_q_optim.zero_grad()
        loss_critic.backward()
        self.cost_q_optim.step()
        stats_critic = {"loss/cost_q_loss": loss_critic.item()}

        return stats_critic

    def actor_loss(self, observations, actions, adv_r, adv_c, step=0):
        exp_adv_r = torch.exp(self.beta * adv_r.detach()).clamp(max=EXP_ADV_MAX)
        exp_adv_c = torch.exp(self.beta_cost * adv_c.detach()).clamp(max=EXP_ADVC_MAX)
        loss_actor = 0    
        all_head_policy_out = self.actor(observations)
        
        for ind, head_policy_out in enumerate(all_head_policy_out):
            if isinstance(head_policy_out, torch.distributions.Distribution):
                head_bc_losses = -head_policy_out.log_prob(actions).sum(-1, keepdim=False)
            elif torch.is_tensor(head_policy_out):
                if head_policy_out.shape != actions.shape:
                    raise RuntimeError("Actions shape missmatch")
                head_bc_losses = torch.sum((head_policy_out - actions) ** 2, dim=1)
            else:
                raise NotImplementedError
            if ind==0:
                loss_head = torch.mean(exp_adv_r * head_bc_losses)
            elif ind==self.num_heads-1: 
                loss_head = torch.mean(exp_adv_c * head_bc_losses)
            else:
                coef = ind/((self.num_heads-1)/2)
                exp_adv = exp_adv_r + coef * exp_adv_c
                loss_head = torch.mean(exp_adv * head_bc_losses)
            
            loss_actor += loss_head/self.num_heads

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        self.actor_lr_schedule.step()

        stats_actor = {"loss/actor_loss": loss_actor.item(),
                      }
        return stats_actor

    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.reward_q_target, self.reward_q, self.tau)
        self._soft_update(self.cost_q_target, self.cost_q, self.tau)

    def setup_optimizers(self, actor_lr, q_lr, value_lr, max_steps):        
        self.reward_value_optim = torch.optim.Adam(self.reward_v.parameters(), lr=value_lr)
        self.cost_value_optim = torch.optim.Adam(self.cost_v.parameters(), lr=value_lr)

        self.reward_q_optim = torch.optim.Adam(self.reward_q.parameters(), lr=q_lr)
        self.cost_q_optim = torch.optim.Adam(self.cost_q.parameters(), lr=q_lr)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optim, max_steps)



class CapsIQLTrainer:
    """
    Multi Head IQL Trainer
    """
    def __init__(
            self,
            model: CapsIQL,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 3e-4,
            q_lr: float = 3e-4,
            value_lr: float = 3e-4,
            max_steps: int = 1000000,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            device="cpu") -> None:

        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimizers(actor_lr, q_lr, value_lr, max_steps)

    def train_one_step(self, observations, next_observations, actions, rewards, costs,
                       done, step):
        
        with torch.no_grad():
            next_vr = self.model.reward_v(next_observations)
            next_vc = self.model.cost_v(next_observations)
        
        # Update Value function
        adv_r, stats_value_r = self.model.value_loss(observations, actions)
        adv_c, stats_value_c = self.model.cost_value_loss(observations, actions)

        # Update Q function
        stats_reward_q = self.model.reward_q_loss(observations, actions, rewards, done, next_vr)
        stats_cost_q = self.model.cost_q_loss(observations, actions, costs, done, next_vc)
        
        # Update actor
        stats_actor = self.model.actor_loss(observations, actions, adv_r, adv_c, step=step)

        self.model.sync_weight()

        # self.logger.store(**stats_vae)
        self.logger.store(**stats_value_r)
        self.logger.store(**stats_value_c)
        self.logger.store(**stats_reward_q)
        self.logger.store(**stats_cost_q)
        self.logger.store(**stats_actor)

    def evaluate(self, eval_episodes, head=0):
        """
        Evaluates the performance of head on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(head=head)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)

        self.model.train()
        agent_performance = (np.mean(episode_rets) / self.reward_scale, \
            np.mean(episode_costs) / self.cost_scale, np.mean(episode_lens))
        
        return agent_performance
    
    def evaluate_switch(self, cost_limit, eval_episodes):
        """
        Evaluates the performance of switching on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout_switch(cost_limit)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)

        self.model.train()
        agent_performance = (np.mean(episode_rets) / self.reward_scale, \
            np.mean(episode_costs) / self.cost_scale, np.mean(episode_lens))
        
        return agent_performance
    

    @torch.no_grad()
    def select_head_q(self, obs, cost_limit, return_id: bool = False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).repeat(self.model.num_heads, 1)
        all_actions = np.array(self.model.actor.act(obs, device=self.device))
        all_act_tensor = torch.tensor(all_actions, dtype=torch.float32).to(self.device)
        all_qc = - self.model.cost_q(obs_tensor, all_act_tensor)
        c_lim_indices = all_qc < cost_limit

        if sum(c_lim_indices)>0:
            all_qr = self.model.reward_q(obs_tensor, all_act_tensor)
            best_r = c_lim_indices.nonzero()[all_qr[c_lim_indices].argmax()]
            id_act = best_r.item()
            action = all_actions[best_r.item()]
        else:
            id_act = all_qc.argmin().item()
            action = all_actions[all_qc.argmin().item()]
        
        if return_id:
            return id_act, action
        else:
            return action
  

    @torch.no_grad()
    def rollout(self, head=0):
        """
        Evaluates the performance of the model on a single episode.
        """
        obs, info = self.env.reset()
        episode_ret, episode_len, episode_cost = 0.0, 0.0, 0
        for _ in range(self.model.episode_len):
            action = self.model.actor.act(obs, headtag=head, device=self.device)
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            cost = info["cost"] * self.cost_scale
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        
        return episode_ret, episode_len, episode_cost
    
    
    @torch.no_grad()
    def rollout_switch(self, cost_limit):
        """
        Evaluates the performance of the model on a single episode.
        """
        obs, info = self.env.reset()
        episode_ret, episode_len, episode_cost = 0.0, 0.0, 0.0
        for step in range(self.model.episode_len):
            disc_cost_limit = max(cost_limit,0) * (1 - self.model.gamma**(self.model.episode_len-step)) / (1 - self.model.gamma) / (self.model.episode_len-step)
            act_id, action = self.select_head_q(obs, disc_cost_limit, return_id=True)

            obs_next, reward, terminated, truncated, info = self.env.step(action)
            cost = info["cost"] * self.cost_scale
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            cost_limit -= cost
            if terminated or truncated:
                break
        
        return episode_ret, episode_len, episode_cost
    
