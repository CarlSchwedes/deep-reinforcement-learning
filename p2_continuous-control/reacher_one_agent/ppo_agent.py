import numpy as np
import random
from collections import namedtuple, deque

from model import DeepDeterministicPolicyGradientNetwork, DeepDeterministicPolicyGradientNetworkWithTD3AndDecayingNoise, SoftActorCriticNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR

# BUFFER_SIZE = int(4e4)      # replay buffer size
# BATCH_SIZE = 64             # minibatch size
# GAMMA = 0.99                # discount factor
# TAU = 1e-3                  # for soft update of target parameters
# LR = 2e-4                   # learning rate
# UPDATE_EVERY = 4            # how often to update the network
# REPLAY_START_SIZE = 64     # warmup before learning from replay


class Agent():
    """Unified Off-Policy Agent supporting DDPG, TD3, SAC + Uniform/Prioritized Replay."""

    def __init__(
        self,
        state_size,
        action_size,
        algorithm="TD3",               # "DDPG", "TD3", or "SAC"
        use_prioritized_replay=False,  # True -> PrioritizedReplayBuffer, False -> ReplayBuffer
        buffer_size=int(4e4),
        batch_size=128,
        gamma=0.99,
        tau=5e-3,
        lr_actor=1e-4,
        lr_critic=1e-3,
        lr_step_size=200,
        lr_gamma=0.95,
        update_every=20,
        num_updates=10,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        per_alpha=0.6,
        beta_start=0.4,
        beta_frames=10000,
        device="cpu",
    ):
        self.algorithm = algorithm.upper()
        self.use_prioritized_replay = use_prioritized_replay
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.num_updates = num_updates
        self.policy_delay = policy_delay if self.algorithm == "TD3" else 1
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.device = torch.device(device)

        # Instantiate Networks directly based on chosen algorithm
        if self.algorithm == "DDPG":
            self.local_net = DeepDeterministicPolicyGradientNetwork(state_size, action_size).to(self.device)
            self.target_net = DeepDeterministicPolicyGradientNetwork(state_size, action_size).to(self.device)
        elif self.algorithm == "TD3":
            self.local_net = DeepDeterministicPolicyGradientNetworkWithTD3AndDecayingNoise(
                state_size, action_size
            ).to(self.device)
            self.target_net = DeepDeterministicPolicyGradientNetworkWithTD3AndDecayingNoise(
                state_size, action_size
            ).to(self.device)
        elif self.algorithm == "SAC":
            self.local_net = SoftActorCriticNetwork(state_size, action_size).to(self.device)
            self.target_net = SoftActorCriticNetwork(state_size, action_size).to(self.device)
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from 'DDPG', 'TD3', or 'SAC'.")

        self.target_net.load_state_dict(self.local_net.state_dict())

        # Group Actor vs Critic parameters dynamically
        actor_params = [p for name, p in self.local_net.named_parameters() if name.startswith("actor")]
        critic_params = [p for name, p in self.local_net.named_parameters() if name.startswith("critic")]

        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)

        # Learning Rate Schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=lr_step_size, gamma=lr_gamma
        )

        # Automatic Entropy Tuning (SAC only)
        if self.algorithm == "SAC":
            self.target_entropy = -float(action_size)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
            self.alpha_scheduler = optim.lr_scheduler.StepLR(
                self.alpha_optimizer, step_size=lr_step_size, gamma=lr_gamma
            )

        # Choose between Replay Memory and Prioritized Replay Memory
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                action_size, buffer_size, batch_size, alpha=per_alpha, device=self.device
            )
        else:
            self.memory = ReplayBuffer(
                action_size, buffer_size, batch_size, device=self.device
            )

        self.t_step = 0
        self.learn_step = 0

    def lr_step(self):
        """Steps the learning rate schedulers down and returns the new LRs."""
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        if self.algorithm == "SAC":
            self.alpha_scheduler.step()

        # Return current actor and critic LRs so you can track and log them
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']
        return actor_lr, critic_lr

    def act(self, state, add_noise=True):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_net.eval()
        with torch.no_grad():
            if self.algorithm == "SAC":
                # Assumes SAC network has sample_action(state, deterministic=not add_noise)
                action, _ = self.local_net.sample_action(state_t, deterministic=not add_noise)
            else:
                action = self.local_net.act(state_t, add_noise=add_noise)
        self.local_net.train()
        return action.cpu().numpy().squeeze(0)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if done and hasattr(self.local_net, "decay_noise"):
            self.local_net.decay_noise()

        last_loss, last_q = None, None
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            self.beta = min(1.0, self.beta + self.beta_increment)
            for _ in range(self.num_updates):
                experiences = self.memory.sample(beta=self.beta)
                last_loss, last_q = self.learn(experiences)

        return last_loss, last_q

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, indices, weights = experiences
        self.learn_step += 1

        # ---------------- 1. Compute Target Q-Values ---------------- #
        with torch.no_grad():
            if self.algorithm == "DDPG":
                next_actions = self.target_net(next_states)
                q_next = self.target_net.critic_forward(next_states, next_actions)

            elif self.algorithm == "TD3":
                noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
                next_actions = (self.target_net(next_states) + noise).clamp(-1.0, 1.0)
                q1_next = self.target_net.critic_forward(next_states, next_actions)
                q2_next = self.target_net.critic2_forward(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)

            elif self.algorithm == "SAC":
                next_actions, next_log_probs = self.local_net.sample_action(next_states)
                q1_next = self.target_net.critic_forward(next_states, next_actions)
                q2_next = self.target_net.critic2_forward(next_states, next_actions)
                alpha = self.log_alpha.exp()
                q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs

            q_targets = rewards + (self.gamma * q_next * (1 - dones))

        # ---------------- 2. Update Critic(s) & Priorities ---------------- #
        q1_expected = self.local_net.critic_forward(states, actions)
        td_error1 = q1_expected - q_targets

        if self.algorithm in ("TD3", "SAC"):
            q2_expected = self.local_net.critic2_forward(states, actions)
            td_error2 = q2_expected - q_targets
            critic_loss = (weights * (td_error1.pow(2) + td_error2.pow(2))).mean()
            priorities = (0.5 * (td_error1.abs() + td_error2.abs()) + 1e-5).detach().cpu().numpy().squeeze()
        else:  # DDPG
            critic_loss = (weights * td_error1.pow(2)).mean()
            priorities = (td_error1.abs() + 1e-5).detach().cpu().numpy().squeeze()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 1.0)
        self.critic_optimizer.step()

        # Safely updates PER priorities, or acts as a no-op for uniform ReplayBuffer
        self.memory.update_priorities(indices, priorities)

        # ---------------- 3. Update Actor & Target Networks ---------------- #
        if self.learn_step % self.policy_delay == 0:
            if self.algorithm == "SAC":
                actions_pred, log_probs = self.local_net.sample_action(states)
                q1_pred = self.local_net.critic_forward(states, actions_pred)
                q2_pred = self.local_net.critic2_forward(states, actions_pred)
                alpha = self.log_alpha.exp().detach()
                actor_loss = (alpha * log_probs - torch.min(q1_pred, q2_pred)).mean()

                # Auto-tune temperature alpha
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            else:
                actions_pred = self.local_net(states)
                actor_loss = -self.local_net.critic_forward(states, actions_pred).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.local_net, self.target_net, self.tau)

        return critic_loss.item(), q1_expected.mean().item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Standard Uniform Replay Buffer matching your exact __init__ signature."""

    def __init__(self, action_size, buffer_size, batch_size, device="cpu"):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self, beta=None):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        # Dummy indices and uniform weights of 1.0 so Agent.learn() stays identical
        indices = None
        weights = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, priorities):
        pass

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Simple, tree-free Proportional Prioritized Experience Replay buffer."""

    def __init__(self, action_size, buffer_size, batch_size, alpha=0.6, device="cpu"):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = torch.device(device)

        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.max_priority = 1.0
        self.eps = 1e-5

    def add(self, state, action, reward, next_state, done):
        """Save a new experience with maximum current priority."""
        e = self.experience(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.memory.append(e)
        self.priorities.append(self.max_priority ** self.alpha)

    def sample(self, beta=0.4):
        """Sample a batch of experiences based on priority probabilities."""
        n_entries = len(self.memory)

        # Convert priorities to sampling probabilities P(i)
        priorities = np.fromiter(self.priorities, dtype=np.float64, count=n_entries)
        probs = priorities / priorities.sum()

        # Sample indices proportional to priority
        indices = np.random.choice(n_entries, size=self.batch_size, p=probs, replace=False)
        experiences = [self.memory[idx] for idx in indices]

        # Compute normalized Importance Sampling (IS) weights
        weights = (n_entries * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Convert to PyTorch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        weights_t = torch.from_numpy(weights).float().unsqueeze(1).to(self.device)

        return (states, actions, rewards, next_states, dones, indices, weights_t)

    def update_priorities(self, indices, td_errors):
        """Update priorities of sampled transitions."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(float(error)) + self.eps) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, abs(float(error)) + self.eps)

    def get_active_priorities(self):
        """Returns a NumPy array of priorities currently containing real experiences."""
        return np.fromiter(self.priorities, dtype=np.float64, count=len(self.memory))

    def __len__(self):
        return len(self.memory)
