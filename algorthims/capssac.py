from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange  # noqa
from osrl.common.net import EnsembleDoubleQCritic, NBHSquashedGaussianMLPActor


class CapsSAC(nn.Module):
    """
    Multi Head SAC Agent

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes 
                               of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes 
                               of the layers in the critic network.
        sample_action_num (int): Number of action samples to draw. 
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks. 
        beta (float): Weight of the KL divergence term.
        num_heads (int): Number of Actor heads.
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 sample_action_num: int = 10,
                 init_temperature: float = 0.1,
                 alphas_betas: list = [0.9, 0.999],
                 actor_betas: list = [0.9, 0.999],
                 critic_betas: list = [0.9, 0.999],
                 BC: bool = True,
                 learnable_temperature: bool = True,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 beta: float = 1.5,
                 num_heads: int = 11,
                 num_q: int = 1,
                 num_qc: int = 1,
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.num_heads = num_heads
        self.num_q = num_q
        self.num_qc = num_qc
        self.sample_action_num = sample_action_num

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = self.action_dim * 2
        self.episode_len = episode_len
        self.max_action = max_action
        self.init_temperature = init_temperature
        self.alphas_betas = alphas_betas
        self.actor_betas = actor_betas
        self.critic_betas = critic_betas
        self.BC = BC
        self.learnable_temperature = learnable_temperature
        self.device = device

        self.actor = NBHSquashedGaussianMLPActor(self.state_dim, self.action_dim,
                                                 self.num_heads,
                                                 self.a_hidden_sizes,
                                                 nn.ReLU).to(self.device)
        self.critic = EnsembleDoubleQCritic(self.state_dim,
                                      self.action_dim,
                                      self.c_hidden_sizes,
                                      nn.ReLU,
                                      num_q=self.num_q,).to(self.device)
        self.cost_critic = EnsembleDoubleQCritic(self.state_dim,
                                           self.action_dim,
                                           self.c_hidden_sizes,
                                           nn.ReLU,
                                           num_q=self.num_qc, ).to(self.device)
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        self.target_entropy = -float(self.action_dim)

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.critic_old = deepcopy(self.critic)
        self.critic_old.eval()
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()


    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module 
        towards the parameters of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _actor_forward(self,
                       obs: torch.tensor,
                       headtag: int,
                       deterministic: bool = False,
                       with_logprob: bool = True):
        """
        Return action distribution and action log prob [optional].
        """
        action, logp = self.actor(obs, headtag, deterministic, with_logprob)
        return action * self.max_action, logp

    def critic_loss(self, observations, next_observations, actions, rewards, done):
        _, _, q1_list, q2_list = self.critic.predict(observations, actions)
        # Bellman backup for Q functions
        with torch.no_grad():
            all_backups = []
            all_actions = []
            all_log_pis = []
            for k in range(self.num_heads):
                next_actions, log_pi = self._actor_forward(next_observations, k, False, True)
                all_actions.append(next_actions)
                all_log_pis.append(log_pi)
            
            all_obs_tensor = torch.vstack([next_observations]*self.num_heads)
            all_act_tensor = torch.vstack(all_actions)
            all_q1_targ, all_q2_targ = self.critic_old.vect_predict(all_obs_tensor, all_act_tensor, num_heads=self.num_heads)
            for k in range(self.num_heads):
                q_targ = torch.min(all_q1_targ[k], all_q2_targ[k])
                backup = rewards + self.gamma * (1 - done) * q_targ - self.alpha.detach() * all_log_pis[k]
                all_backups.append(backup)

        # MSE loss against Bellman backup
        loss_critic = 0
        for k in range(self.num_heads):
            backup = all_backups[k]
            loss_critic += (self.critic.loss(backup, q1_list) + self.critic.loss(backup, q2_list)) / self.num_heads

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        stats_critic = {"loss/critic_loss": loss_critic.item()}
        return loss_critic, stats_critic

    def cost_critic_loss(self, observations, next_observations, actions, costs, done):
        _, _, qc1_list, qc2_list = self.cost_critic.predict(observations, actions)
        costs= -costs
        # Bellman backup for Q functions
        with torch.no_grad():
            # all_backups = {}
            all_backups = []
            all_actions = []
            all_log_pis = []
            for k in range(self.num_heads):
                next_actions_cost, log_pi_cost = self._actor_forward(next_observations, k, False, True)
                all_actions.append(next_actions_cost)
                all_log_pis.append(log_pi_cost)

            all_obs_tensor = torch.vstack([next_observations]*self.num_heads)
            all_act_tensor = torch.vstack(all_actions)
            all_q1_targ, all_q2_targ = self.cost_critic_old.vect_predict(all_obs_tensor, all_act_tensor, num_heads=self.num_heads)
    
            for k in range(self.num_heads):
                q_targ = torch.min(all_q1_targ[k], all_q2_targ[k])
                backup = costs + self.gamma * (1 - done) * q_targ - self.alpha.detach() * all_log_pis[k]
                all_backups.append(backup)

        # MSE loss against Bellman backup
        loss_cost_critic = 0
        for k in range(self.num_heads):
            # backup = all_backups[f"head_{k}"]
            backup = all_backups[k]
            loss_cost_critic += (self.cost_critic.loss(backup, qc1_list) + self.cost_critic.loss(backup, qc2_list)) / self.num_heads

        self.cost_critic_optim.zero_grad()
        loss_cost_critic.backward()
        self.cost_critic_optim.step()
        stats_cost_critic = {"loss/cost_critic_loss": loss_cost_critic.item()}
        return loss_cost_critic, stats_cost_critic
    

    def alpha_loss(self, observations):
        with torch.no_grad():
            heads_log_pi_list = []
            for k in range(self.num_heads):
                _, logp_pi = self._actor_forward(observations, k, False, True)
                heads_log_pi_list.add(logp_pi)

        alpha_loss = 0 
        for k in range(self.num_heads):
            alpha_loss += -self.log_alpha * (heads_log_pi_list[k] + self.target_entropy)
        alpha_loss = alpha_loss.mean()
        
        self.alpha_loss.zero_grad()
        alpha_loss.backward()
        self.alpha_loss.step()
        stats_alpha = {"loss/alpha_loss": alpha_loss.item()}

        return alpha_loss, stats_alpha
    

    def actor_loss(self, observations, actions):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False
    
        loss_actor = 0
        loss_alpha = 0
        self.log_alpha_optim.zero_grad()
        for k in range(self.num_heads):
            agent_actions, log_pi = self._actor_forward(observations, k, False, True)
            q1_pi, q2_pi, _, _ = self.critic.predict(observations, agent_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            qc1_pi, qc2_pi, _, _ = self.cost_critic.predict(observations, agent_actions)
            qc_pi = torch.min(qc1_pi, qc2_pi)
            if k < self.num_heads-1:
                coef = k/((self.num_heads-1)/2)
                q_pi_mix = q_pi + coef * qc_pi
            else:
                q_pi_mix = qc_pi
            
            # normalizing the q values helps stabilize learning following:A Minimalist Approach to Offline Reinforcement Learning 
            # (https://arxiv.org/abs/2106.06860)
            bc_lmbda = 2.5 / q_pi_mix.abs().mean().detach()
            loss_head = (self.alpha.detach() * log_pi - bc_lmbda * q_pi_mix.mean() + F.mse_loss(agent_actions, actions)).mean()
            loss_actor += loss_head
            if self.learnable_temperature:
                loss_alpha_head = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
                loss_alpha += loss_alpha_head

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        if self.learnable_temperature:
            loss_alpha.backward()
            self.log_alpha_optim.step()
        
        stats_actor = {"loss/actor_loss": loss_actor.item(),
                       "loss/alpha_loss": loss_alpha.item(),
                       "loss/alpha_value": self.alpha.item()}

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        return loss_actor, stats_actor

    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def setup_optimizers(self, actor_lr, critic_lr, alpha_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=self.actor_betas)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=self.critic_betas)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(),
                                                  lr=critic_lr, betas=self.critic_betas)
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=self.alphas_betas)

    def act(self,
            obs: np.ndarray,
            headtag: int,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single obs, return the action, logp.
        """
        obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        action, logp_a = self._actor_forward(obs, headtag, deterministic, with_logprob)
        action = action.data.numpy() if self.device == "cpu" else action.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu(
        ).numpy()
        
        return np.squeeze(action, axis=0), np.squeeze(logp_a)


class CapsSACTrainer:
    """
    Multi-Head SAC Trainer
    
    Args:
        model (CapsSAC): The CapsSAC model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic
        alpha_lr (float): learning rate for alpha
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """

    def __init__(
            self,
            model: CapsSAC,
            env: gym.Env,
            logger: WandbLogger = DummyLogger(),
            # training params
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-4,
            alpha_lr: float = 1e-4,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            device="cpu") -> None:

        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.model.setup_optimizers(actor_lr, critic_lr, alpha_lr)

    def train_one_step(self, observations, next_observations, actions, rewards, costs,
                       done):
        # update critic
        loss_critic, stats_critic = self.model.critic_loss(observations,
                                                           next_observations, actions,
                                                           rewards, done)
        # update cost critic
        loss_cost_critic, stats_cost_critic = self.model.cost_critic_loss(
            observations, next_observations, actions, costs, done)
        
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)

        self.model.sync_weight()

        self.logger.store(**stats_critic)
        self.logger.store(**stats_cost_critic)
        self.logger.store(**stats_actor)

    def evaluate(self, headtag, eval_episodes):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        all_cost_estimates = []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(headtag)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)

        self.model.train()
        agent_performance = (np.mean(episode_rets) / self.reward_scale, \
            np.mean(episode_costs) / self.cost_scale, np.mean(episode_lens))
        
        return agent_performance
    

    def evaluate_head_switch(self, cost_limit, eval_episodes):
        """
        Evaluates the performance of the model on a number of episodes.
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
    def select_head(self, obs, cost_limit):
        actions_dict = {}
        max_reward = -np.inf 
        min_cost = np.inf
        best_action_index = 0
        best_cost_index = 0
        cost_limit_staisfied = False
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        for headtag in range(self.model.num_heads):
            action, _ = self.model.act(obs, headtag, True, True)
            actions_dict[f"act_{headtag}"] = action
            act_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
            qc1_targ, qc2_targ, _, _ = self.model.cost_critic.predict(obs_tensor, act_tensor)
            qc_targ = - torch.min(qc1_targ, qc2_targ) 
            
            if (max(qc_targ.item(),0)) < cost_limit:
                cost_limit_staisfied = True
                q1_targ, q2_targ, _, _ = self.model.critic.predict(obs_tensor, act_tensor)
                q_targ = torch.min(q1_targ, q2_targ) 
                
                if q_targ.item() > max_reward:
                    best_action_index = headtag
                    max_reward = q_targ.item()
            
            elif qc_targ.item() < min_cost:
                best_cost_index = headtag
                min_cost = qc_targ.item()
        if cost_limit_staisfied:
            action = actions_dict[f"act_{best_action_index}"]
            return best_action_index, action
        else:
            action = actions_dict[f"act_{best_cost_index}"]
            return best_cost_index, action
     

    @torch.no_grad()
    def rollout(self, headtag):
        """
        Evaluates the performance of the model on a single episode.
        """
        obs, info = self.env.reset()
        episode_ret, episode_len, episode_cost = 0.0, 0.0, 0        
        for _ in range(self.model.episode_len):
            action, _ = self.model.act(obs, headtag, True, True)
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
        all_action_ids_sat, all_action_ids_nonsat = [], []
        for step in range(self.model.episode_len):
            disc_cost_limit = max(cost_limit,0) * (1 - self.model.gamma**(self.model.episode_len-step)) / (1 - self.model.gamma) / (self.model.episode_len-step)
            act_id, action = self.select_head(obs, disc_cost_limit)
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

