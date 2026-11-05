import copy

import numpy as np
import torch
import torch.nn.functional as F

import rl_utils
from networks import PolicyNet, ValueNet


class TRPO:
    """Trust Region Policy Optimization agent。"""

    def __init__(self, state_dim, action_dim, hidden_dim, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.device = device

    def take_action(self, state):
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action_dist = torch.distributions.Categorical(self.actor(state))
        return action_dist.sample().item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        """计算平均KL散度的Hessian与vector的乘积。"""
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.distributions.kl.kl_divergence(
            old_action_dists, new_action_dists
        ).mean()
        kl_grad = torch.autograd.grad(
            kl, self.actor.parameters(), create_graph=True
        )
        kl_grad_vector = torch.cat([grad.reshape(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(
            kl_grad_vector_product, self.actor.parameters()
        )
        return torch.cat([grad.reshape(-1) for grad in grad2]).detach()

    def conjugate_gradient(self, grad, states, old_action_dists):
        """使用共轭梯度法近似求解H^{-1}g。"""
        x = torch.zeros_like(grad)
        residual = grad.clone()
        direction = grad.clone()
        residual_norm = torch.dot(residual, residual)

        for _ in range(10):
            hessian_direction = self.hessian_matrix_vector_product(
                states, old_action_dists, direction
            )
            denominator = torch.dot(direction, hessian_direction)
            step = residual_norm / (denominator + 1e-8)
            x += step * direction
            residual -= step * hessian_direction
            new_residual_norm = torch.dot(residual, residual)
            if new_residual_norm < 1e-10:
                break
            beta = new_residual_norm / (residual_norm + 1e-8)
            direction = residual + beta * direction
            residual_norm = new_residual_norm
        return x

    @staticmethod
    def _log_probs(actor, states, actions):
        action_dist = torch.distributions.Categorical(actor(states))
        return action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def compute_surrogate_obj(self, states, actions, advantage,
                              old_log_probs, actor):
        log_probs = self._log_probs(actor, states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vector):
        old_parameters = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        ).detach()
        old_objective = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        ).detach()

        for index in range(15):
            candidate_parameters = old_parameters + self.alpha**index * max_vector
            candidate_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                candidate_parameters, candidate_actor.parameters()
            )
            candidate_dists = torch.distributions.Categorical(
                candidate_actor(states)
            )
            kl_divergence = torch.distributions.kl.kl_divergence(
                old_action_dists, candidate_dists
            ).mean()
            objective = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, candidate_actor
            )
            if objective > old_objective and kl_divergence < self.kl_constraint:
                return candidate_parameters
        return old_parameters

    def policy_learn(self, states, actions, old_action_dists,
                     old_log_probs, advantage):
        surrogate_objective = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )
        gradients = torch.autograd.grad(
            surrogate_objective, self.actor.parameters()
        )
        objective_gradient = torch.cat(
            [gradient.reshape(-1) for gradient in gradients]
        ).detach()
        descent_direction = self.conjugate_gradient(
            objective_gradient, states, old_action_dists
        )
        hessian_direction = self.hessian_matrix_vector_product(
            states, old_action_dists, descent_direction
        )
        scale = torch.sqrt(
            2 * self.kl_constraint /
            (torch.dot(descent_direction, hessian_direction) + 1e-8)
        )
        new_parameters = self.line_search(
            states, actions, advantage, old_log_probs, old_action_dists,
            descent_direction * scale
        )
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_parameters, self.actor.parameters()
        )

    def update(self, transition_dict):
        states = torch.as_tensor(
            np.asarray(transition_dict['states']),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            transition_dict['actions'], dtype=torch.long, device=self.device
        ).view(-1, 1)
        rewards = torch.as_tensor(
            transition_dict['rewards'],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        next_states = torch.as_tensor(
            np.asarray(transition_dict['next_states']),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            transition_dict['dones'],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(
            self.gamma, self.lmbda, td_delta
        ).to(self.device).unsqueeze(1)

        old_log_probs = self._log_probs(
            self.actor, states, actions
        ).detach()
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach()
        )
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.policy_learn(
            states, actions, old_action_dists, old_log_probs, advantage
        )
