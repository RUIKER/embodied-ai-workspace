import copy

import numpy as np
import torch
import torch.nn.functional as F

import rl_utils
from networks import PolicyNetContinuous, ValueNet


class TRPOContinuous:
    """连续动作版本的TRPO agent。"""

    def __init__(self, state_dim, action_dim, hidden_dim, action_low,
                 action_high, lmbda, kl_constraint, alpha, critic_lr, gamma,
                 reward_shift, reward_scale, device):
        self.actor = PolicyNetContinuous(
            state_dim, hidden_dim, action_dim, action_low, action_high
        ).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.action_low = torch.as_tensor(
            action_low, dtype=torch.float32, device=device
        )
        self.action_high = torch.as_tensor(
            action_high, dtype=torch.float32, device=device
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.device = device

    def _distribution(self, actor, states):
        mean, std = actor(states)
        return torch.distributions.Normal(mean, std)

    def take_action(self, state):
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action = self._distribution(self.actor, state).sample()
        action = torch.clamp(action, self.action_low, self.action_high)
        return action.squeeze(0).detach().cpu().numpy()

    def hessian_matrix_vector_product(self, states, old_action_dists,
                                      vector, damping=0.1):
        new_action_dists = self._distribution(self.actor, states)
        kl = torch.distributions.kl.kl_divergence(
            old_action_dists, new_action_dists
        ).sum(dim=-1).mean()
        kl_grad = torch.autograd.grad(
            kl, self.actor.parameters(), create_graph=True
        )
        kl_grad_vector = torch.cat([grad.reshape(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(
            kl_grad_vector_product, self.actor.parameters()
        )
        hessian_vector = torch.cat([grad.reshape(-1) for grad in grad2])
        return hessian_vector.detach() + damping * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        residual = grad.clone()
        direction = grad.clone()
        residual_norm = torch.dot(residual, residual)
        for _ in range(10):
            hessian_direction = self.hessian_matrix_vector_product(
                states, old_action_dists, direction
            )
            step = residual_norm / (
                torch.dot(direction, hessian_direction) + 1e-8
            )
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
        mean, std = actor(states)
        return torch.distributions.Normal(mean, std).log_prob(actions).sum(
            dim=-1, keepdim=True
        )

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
            candidate_dists = self._distribution(candidate_actor, states)
            kl_divergence = torch.distributions.kl.kl_divergence(
                old_action_dists, candidate_dists
            ).sum(dim=-1).mean()
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
        curvature = torch.dot(descent_direction, hessian_direction)
        scale = torch.sqrt(
            2 * self.kl_constraint / torch.clamp(curvature, min=1e-8)
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
            np.asarray(transition_dict['actions']),
            dtype=torch.float32,
            device=self.device,
        )
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

        scaled_rewards = (rewards + self.reward_shift) / self.reward_scale
        td_target = scaled_rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(
            self.gamma, self.lmbda, td_delta
        ).to(self.device).unsqueeze(1)

        old_log_probs = self._log_probs(
            self.actor, states, actions
        ).detach()
        old_action_dists = self._distribution(self.actor, states)
        old_action_dists = torch.distributions.Normal(
            old_action_dists.loc.detach(), old_action_dists.scale.detach()
        )
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.policy_learn(
            states, actions, old_action_dists, old_log_probs, advantage
        )
