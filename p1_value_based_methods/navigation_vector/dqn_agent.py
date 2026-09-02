import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork, DistributionalDuelingNoisyQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR

BUFFER_SIZE = int(4e4)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 2e-4                   # learning rate
UPDATE_EVERY = 4            # how often to update the network
REPLAY_START_SIZE = 64     # warmup before learning from replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, 
                 state_size, 
                 action_size, 
                 use_prioritized_replay=False, 
                 use_ddqn_dueling_network=False, 
                 use_replay_start_size=False, 
                 use_noisy_nets=False, 
                 n_steps=1,
                 use_distributional_rl=False
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            use_prioritized_replay (bool): whether to use Prioritized Experience Replay
            use_ddqn_dueling_network (bool): whether to use Double DQN with Dueling Networks
            use_replay_start_size (bool): whether to use a warmup period before learning from replay
            use_noisy_nets (bool): whether to use Noisy Networks
            n_steps (int): number of steps for n-step returns (default is 1 for standard DQN)
            use_distributional_rl (bool): whether to use Distributional RL
        """
        self.state_size = state_size
        self.action_size = action_size
        self.use_prioritized_replay = use_prioritized_replay
        self.use_ddqn_dueling_network = use_ddqn_dueling_network
        self.use_replay_start_size = use_replay_start_size
        self.use_noisy_nets = use_noisy_nets
        self.use_distributional_rl = use_distributional_rl

        if self.use_replay_start_size:
            self.replay_start_size = REPLAY_START_SIZE if self.use_replay_start_size else BATCH_SIZE

        if self.use_distributional_rl:
            # For Categorical Distributional RL (C51)
            self.num_atoms = 51
            self.v_min = -10.0
            self.v_max = 10.0
            # Create the static support vector tensor: [-10.0, -9.6, -9.2, ..., +10.0]
            self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)

            self.qnetwork_local = DistributionalDuelingNoisyQNetwork(state_size, action_size, self.num_atoms).to(device)
            self.qnetwork_target = DistributionalDuelingNoisyQNetwork(state_size, action_size, self.num_atoms).to(device)
        else:
            if self.use_ddqn_dueling_network:
                self.qnetwork_local = DuelingQNetwork(state_size, action_size, use_noisy_nets=self.use_noisy_nets).to(device)
                self.qnetwork_target = DuelingQNetwork(state_size, action_size, use_noisy_nets=self.use_noisy_nets).to(device)
            else:
                self.qnetwork_local = QNetwork(state_size, action_size, use_noisy_nets=self.use_noisy_nets).to(device)
                self.qnetwork_target = QNetwork(state_size, action_size, use_noisy_nets=self.use_noisy_nets).to(device)

        self.n_steps = n_steps

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # decay_rate=0.995 means LR drops by 0.5% every time step() is called on it
        self.scheduler = StepLR(self.optimizer, step_size=200, gamma=0.8) # ExponentialLR(self.optimizer, gamma=0.998)

        # Chose between Replay Memory and Prioritized Replay Memory
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # N-step return tracking
        self.n_step_buffer = deque(maxlen=n_steps)  # Store n-step experiences
        self.gamma_power = GAMMA ** n_steps  # Gamma^n for n-step returns

    def lr_step(self):
        """Steps the learning rate scheduler down and returns the new LR."""
        self.scheduler.step()
        # Return the current LR so we can track and log it
        return self.optimizer.param_groups[0]['lr']

    def step(self, state, action, reward, next_state, done):
        # 1. add current transition to temporary n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # 2. If the episode is NOT done, process only when we have a full rolling window
        if not done:
            if len(self.n_step_buffer) == self.n_steps:
                n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()
                self.memory.add(n_state, n_action, n_reward, n_next_state, n_done)
        else:
            # 3. If the episode IS done, cleanly flush out all remaining steps in order
            while len(self.n_step_buffer) > 0:
                n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()
                self.memory.add(n_state, n_action, n_reward, n_next_state, n_done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_start_size:
                # IMPORTANT: We must pass GAMMA^(n_steps) to the learn() function!
                gamma_n = GAMMA ** self.n_steps

                if hasattr(self.qnetwork_local, 'reset_noise'):
                    self.qnetwork_local.reset_noise()
                if hasattr(self.qnetwork_target, 'reset_noise'):
                    self.qnetwork_target.reset_noise()

                if self.use_prioritized_replay:
                    experiences, indices, is_weights = self.memory.sample()
                    loss, avg_q = self.learn(experiences, indices, is_weights, gamma_n)
                else:
                    experiences = self.memory.sample()
                    loss, avg_q = self.learn(experiences, None, None, gamma_n)
                
                return loss, avg_q
        # in case no update is performed for this iteration, return None for loss and avg_q
        return None, None

    def _get_n_step_info(self):
        """Computes the discounted n-step rewards and the final next state."""
        # Get the oldest element (starting point of the n-steps)
        state, action, reward, next_state, done = self.n_step_buffer[0]
        
        # Initialize discounted reward
        discounted_reward = reward
        
        # Go through the subsequent steps in the short-term buffer
        for i in range(1, len(self.n_step_buffer)):
            _, _, r, next_s, d = self.n_step_buffer[i]
            
            # R_t = R_t + GAMMA^i * R_{t+i}
            discounted_reward += (GAMMA ** i) * r
            
            # The final next state shifts backward
            next_state = next_s
            done = d
            
            # If a real terminal end occurs in the middle, we break
            if done:
                break
                
        # We only remove the oldest element from the Deque (FIFO)
        # If an episode was over (done=True), the while loop in step() empties the rest.
        if not done:
            self.n_step_buffer.popleft()
        else:
            self.n_step_buffer.clear()
            
        return state, action, discounted_reward, next_state, done

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
            if not self.use_distributional_rl:
                q_values = self.qnetwork_local(state)
            else:
                # Get probability distributions shape: (1, action_size, 51)
                probs = self.qnetwork_local(state)
                # Calculate expected values: Q(s,a) = sum(probabilities * atom_values)
                q_values = (probs * self.support).sum(dim=-1) # Shape: (1, action_size)
        self.qnetwork_local.train()

        # If using Noisy Networks, the architecture handles exploration internally via weights.
        # We completely bypass random sampling and act deterministically on the noisy outputs.
        if hasattr(self.qnetwork_local, 'reset_noise'):
            return np.argmax(q_values .cpu().data.numpy())

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(q_values .cpu().data.numpy())
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
        batch_size = states.size(0)

        # =========================================================================
        # PATHWAY A: CATEGORICAL / DISTRIBUTIONAL DQN (C51 / RAINBOW)
        # =========================================================================
        if self.use_distributional_rl:
            with torch.no_grad():
                # 1. Fetch upcoming probability matrices from target network (Batch, Actions, Atoms)
                next_probs = self.qnetwork_target(next_states) 
                
                # 2. DDQN Selection: Find the best action using the LOCAL network expected values
                # Q(s,a) = sum(probabilities * support_values)
                next_q_values = (self.qnetwork_local(next_states) * self.support).sum(dim=-1)
                best_actions_next = next_q_values.max(1)[1].unsqueeze(1).unsqueeze(2) 
                best_actions_next = best_actions_next.expand(batch_size, 1, self.num_atoms)
                
                # 3. Gather target distribution matching the chosen actions
                next_probs_best = next_probs.gather(1, best_actions_next).squeeze(1) # (Batch, Atoms)

                # 4. Perform Bellman Projection for Distributions: Tz = r + gamma^n * z
                delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
                Tz = rewards + (gamma * self.support.unsqueeze(0) * (1 - dones))
                Tz = Tz.clamp(min=self.v_min, max=self.v_max)
                
                # Calculate corresponding histogram bin boundaries
                b = (Tz - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                # Fix clipping boundaries if target calculations land exactly on whole bounds
                l[(u == l) & (l > 0)] -= 1
                u[(u == l) & (u < self.num_atoms - 1)] += 1

                # Accumulate values into the target categorical template array (m)
                m = states.new_zeros(batch_size, self.num_atoms)
                offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long().unsqueeze(1).to(device)
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_probs_best * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_probs_best * (b - l.float())).view(-1))

            # 5. Extract current distributions for actions executed from local network
            current_probs = self.qnetwork_local(states)
            actions_expanded = actions.unsqueeze(2).expand(batch_size, 1, self.num_atoms)
            current_probs_taken = current_probs.gather(1, actions_expanded).squeeze(1)

            # Prevent structural log(0) NaN drops via strict lower bound boundary clamping
            current_probs_taken = torch.clamp(current_probs_taken, min=1e-5)

            # Categorical Cross-Entropy works as the absolute TD-error signal profile for PER tracking
            td_errors = -(m * current_probs_taken.log()).sum(dim=-1)
            
            # Loss assignment (Apply importance sampling weights if PER is running)
            if isinstance(self.memory, PrioritizedReplayBuffer):
                loss = (is_weights.squeeze() * td_errors).mean()
                self.memory.update_priorities(indices, td_errors.detach().cpu().numpy().flatten())
            else:
                loss = td_errors.mean()

            # Backprop updates
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

            # Calculate a reconstructed scalar expectation metric for TensorBoard compatibility
            with torch.no_grad():
                avg_q_out = (current_probs_taken * self.support).sum(dim=-1).mean().item()
            return loss.item(), avg_q_out

        # =========================================================================
        # PATHWAY B: STANDARD SCALAR VALUE-BASED DQN (LEGACY RUNTIMES)
        # =========================================================================
        else:
            if self.use_prioritized_replay or self.use_ddqn_dueling_network:
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
    
    def __init__(self, action_size, buffer_size, batch_size, alpha=0.3, beta_start=0.4, beta_frames=100000):
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
        self.beta = min(.9, self.beta_start + self.frame_count * (.9 - self.beta_start) / self.beta_frames)
        
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