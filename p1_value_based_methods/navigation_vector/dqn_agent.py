import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(4e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 2.5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, use_prioritized_replay=False, use_ddqn_dueling_network=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.use_prioritized_replay = use_prioritized_replay
        self.use_ddqn_dueling_network = use_ddqn_dueling_network
        # Q-Network
        if self.use_ddqn_dueling_network:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Chose between Replay Memory and Prioritized Replay Memory
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.use_prioritized_replay:
                    experiences, indices, is_weights = self.memory.sample()
                    loss, avg_q = self.learn(experiences, indices, is_weights, GAMMA)
                else:
                    experiences = self.memory.sample()
                    loss, avg_q = self.learn(experiences, None, None, GAMMA)
                
                return loss, avg_q
        # in case no update is performed for this iteration, return None for loss and avg_q
        return None, None

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection (ignored if using Noisy Networks)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, indices, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            indices (Tuple[int]): indices of experiences in the replay buffer
            is_weights (Tuple[float]): importance-sampling weights for the experiences
            gamma (float): discount factor (GAMMA for 1-step, GAMMA^n for n-step returns)
        """
        states, actions, rewards, next_states, dones = experiences

        if self.use_ddqn_dueling_network:
            # 1. DOUBLE DQN WITH DUELING NETWORKS
            # Use the LOCAL Dueling network to SELECT the best action index for the next states
            # (Extracts the index component [1] from PyTorch's max() function)
            best_actions_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            
            # Get max predicted Q values (for next states) using the TARGET Dueling network
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions_next)
        else:
            # Get max predicted Q values (for next states)
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if isinstance(self.memory, PrioritizedReplayBuffer):
            # Calculate individual TD errors for priority updates
            # We must detach and convert to a flat numpy array
            with torch.no_grad():
                # Form: (Batch_Size, 1)
                td_errors_tensor = torch.abs(Q_targets - Q_expected)
                # Convert to flat NumPy array for the buffer
                td_errors_numpy = td_errors_tensor.cpu().numpy().flatten()

            self.memory.update_priorities(indices, td_errors_numpy)

            # Calculate weighted Mean Squared Error loss using the IS weights
            # We square the differences element-wise, multiply by is_weights, then take the mean
            loss = (is_weights * (Q_expected - Q_targets) ** 2).mean()
        else:
            # Standard MSE loss for uniform sampling
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        return loss.item(), Q_expected.mean().item()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store priority experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # Determines how much prioritization is used (0 = uniform, 1 = full prioritization)
        
        # Importance Sampling (IS) parameters: beta grows linearly over time up to 1.0
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0
        
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = []
        self.pos = 0
        
        # Priority storage matching the maximum capacity of the buffer
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0  # New experiences get max priority to ensure they are sampled at least once

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory. New items receive max_priority."""
        exp = self.experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(exp)
        else:
            self.memory[self.pos] = exp
            
        # Assign maximum priority to new transitions
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self):
        """Samples a batch of experiences based on priorities and returns IS weights."""
        actual_size = len(self.memory)
        prios = self.priorities[:actual_size]
        
        # P(i) = p_i^alpha / sum(p_k^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on the probability distribution
        indices = np.random.choice(actual_size, self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        
        # Linearly anneal Beta towards 1.0
        self.frame_count += 1
        self.beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)
        
        # Compute Importance Sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        weights = (actual_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for numerical stability
        
        # Convert data to PyTorch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        is_weights = torch.from_numpy(np.vstack(weights)).float().to(device)
        
        return (states, actions, rewards, next_states, dones), indices, is_weights

    def update_priorities(self, indices, errors, offset=1e-5):
        """Updates transition priorities using absolute TD errors."""

        if hasattr(errors, "detach"):
            errors = errors.detach().cpu().numpy()

        errors = np.array(errors).flatten()
        
        for idx, error in zip(indices, errors):
            p = np.abs(error) + offset  # offset prevents zero priority
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def get_active_priorities(self):
        """Returns the slice of priorities currently containing real experiences."""
        return self.priorities[:len(self.memory)]

    def __len__(self):
        return len(self.memory)