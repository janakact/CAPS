import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.normal import Normal
import sys

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.

    Args:
        sizes (list): A list of integers specifying the size of each layer in the MLP.
        activation (nn.Module): The activation function to use for all layers except the output layer.
        output_activation (nn.Module): The activation function to use for the output layer. Defaults to nn.Identity.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)

class NBHSquashedGaussianMLPActor(nn.Module):
    '''
    A MLP Gaussian actor with two heads (reward-maximizing, cost-minimizing), can also be used as a deterministic actor with two heads
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
    '''

    def __init__(self, obs_dim, act_dim, num_heads, hidden_sizes, activation):
        super().__init__()
        hidden_sizes = [int(sz) for sz in(hidden_sizes)]
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.tasks = nn.ModuleDict()

        for k in range(num_heads):
            self.tasks[f'a_{k}'] =  nn.Linear(hidden_sizes[-1], act_dim)
            self.tasks[f'log_pi_{k}'] = nn.Linear(hidden_sizes[-1], act_dim) 

    def forward(self,
                obs,
                headtag,
                deterministic=False,
                with_logprob=True,
                with_distribution=False,
                ):
        net_out = self.net(obs)
        
        action_mu = self.tasks[f"a_{headtag}"](net_out)
        log_std = self.tasks[f"log_pi_{headtag}"](net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(action_mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = action_mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
            
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        if with_distribution:
            return pi_action, logp_pi, pi_distribution
        return pi_action, logp_pi

class MHGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        num_heads: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.device =  device
        self.net = mlp([state_dim] + n_hidden * [hidden_dim], nn.ReLU, nn.ReLU)
        self.tasks = nn.ModuleDict() 
        self.log_stds = []
        for k in range(self.num_heads):
            self.tasks[f'a_{k}'] =  nn.Linear(hidden_dim, act_dim)
            self.log_stds.append(nn.Parameter(torch.zeros(act_dim, dtype=torch.float32, device=self.device)))

        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        net_out = self.net(obs)
        all_heads_dist = []
        for head in range(self.num_heads):
            mean_head = torch.tanh(self.tasks[f'a_{head}'](net_out))
            std_head = torch.exp(self.log_stds[head].clamp(LOG_STD_MIN, LOG_STD_MAX))
            all_heads_dist.append(Normal(mean_head, std_head))
        # print("num_heads", self.num_heads)
        # print("all_heads_dist", len(all_heads_dist))
        # sys.exit()
        return all_heads_dist

    @torch.no_grad()
    def act(self, state: np.ndarray, tasktag: int = -1, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        all_actions = []
        all_heads_dist = self(state)
        # print("all_heads_dist", len(all_heads_dist))
        # print("tasktag", tasktag)
        if tasktag > -1 :
            action_head = all_heads_dist[tasktag].mean
            action_head = torch.clamp(self.max_action * action_head, -self.max_action, self.max_action)
            return action_head.cpu().data.numpy().flatten()
        else:
            for head_dist in all_heads_dist:
                action_head = head_dist.mean #if not self.training else dist_reward.sample()
                action_head = torch.clamp(self.max_action * action_head, -self.max_action, self.max_action)
                all_actions.append(action_head.cpu().data.numpy().flatten())
            
            return all_actions
        
    @torch.no_grad()
    def act_batch(self, state: torch.Tensor, tasktag: int = 0, device: str = "cpu"):
        # state = torch.tensor(state, device=device, dtype=torch.float32)
        all_heads_dist = self(state)
        action_head = all_heads_dist[tasktag].mean
        action_head = torch.clamp(self.max_action * action_head, -self.max_action, self.max_action)
        return action_head
       
class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = mlp([state_dim] + n_hidden * [hidden_dim], nn.ReLU, nn.ReLU)
        self.mu_layer_reward = nn.Linear(hidden_dim, act_dim)
        self.mu_layer_cost = nn.Linear(hidden_dim, act_dim)

        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        net_out = self.net(obs)
        action_reward = torch.tanh(self.mu_layer_reward(net_out))
        action_cost = torch.tanh(self.mu_layer_cost(net_out))

        return action_reward, action_cost

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action_reward, action_cost = self(state)
        action_reward = torch.clamp(action_reward * self.max_action, -self.max_action, self.max_action).cpu().data.numpy().flatten()
        action_cost = torch.clamp(action_cost * self.max_action, -self.max_action, self.max_action).cpu().data.numpy().flatten()
        return action_reward, action_cost


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
    
class EnsembleDoubleQCritic(nn.Module):
    '''
    An ensemble of double Q network to address the overestimation issue.
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2, use_layernorm=False):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"

        self.q1_nets = nn.ModuleList([
        mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
        for i in range(num_q)
        ])
        self.q2_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        q1 = [torch.squeeze(q(data), -1) for q in self.q1_nets]
        q2 = [torch.squeeze(q(data), -1) for q in self.q2_nets]
        return q1, q2

    def predict(self, obs, act):
        q1_list, q2_list = self.forward(obs, act)
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        # qs = torch.vstack(q_list)  # [num_q, batch_size]
        qs1_min, qs2_min = torch.min(qs1, dim=0).values, torch.min(qs2, dim=0).values

        return qs1_min, qs2_min, q1_list, q2_list
    
    def vect_predict(self, obs, act, num_heads=1):
        q1_list, q2_list = self.forward(obs, act)
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        q1_min_list = []
        q2_min_list = []
        batch_size = obs.shape[0]//num_heads
        for k in range(num_heads):
            qs1_min = torch.min(qs1[:, k*batch_size:(k+1)*batch_size], dim=0).values
            qs2_min = torch.min(qs2[:, k*batch_size:(k+1)*batch_size], dim=0).values
            q1_min_list.append(qs1_min)
            q2_min_list.append(qs2_min)

        return q1_min_list, q2_min_list

    def loss(self, target, q_list=None):
        losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)
