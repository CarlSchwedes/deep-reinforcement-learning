import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DQNNatureNetwork, DuelingQNetwork, DuelingDQNNatureNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
CNN_LR = 2.5e-4         # Nature DQN-style learning rate for pixel input
CNN_REPLAY_START = 10000  # warmup before learning from replay
PER_ALPHA = 0.6         # Prioritized Experience Replay: priority exponent
PER_BETA = 0.4          # Prioritized Experience Replay: importance-sampling exponent (annealed to 1.0)
PER_EPSILON = 1e-6      # Prioritized Experience Replay: small constant to avoid zero priorities

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, qnetwork_class=QNetwork,
                 lr=None, batch_size=BATCH_SIZE, replay_start_size=None, grad_clip=None,
                 use_prioritized=False, use_double_dqn=False, use_noisy_nets=False,
                 use_dueling=False, n_steps=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            qnetwork_class (type): network class to use for the local/target Q-networks
            lr (float): optimizer learning rate; defaults depend on selected network
            batch_size (int): replay minibatch size
            replay_start_size (int): minimum replay size before updates begin
            grad_clip (float|None): gradient clipping max norm (None disables clipping)
            use_prioritized (bool): whether to use Prioritized Experience Replay
            use_double_dqn (bool): whether to use Double DQN to reduce overestimation
            use_noisy_nets (bool): whether to use Noisy Networks for learned exploration
            use_dueling (bool): whether to use Dueling Network architecture
            n_steps (int): number of steps for n-step returns (1 = standard DQN)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.qnetwork_class = qnetwork_class
        self.network_name = getattr(qnetwork_class, "__name__", str(qnetwork_class))
        self.batch_size = batch_size
        self.is_cnn = qnetwork_class in [DQNNatureNetwork, DuelingDQNNatureNetwork]
        self.n_steps = n_steps

        if lr is None:
            lr = CNN_LR if self.is_cnn else LR
        if replay_start_size is None:
            replay_start_size = CNN_REPLAY_START if self.is_cnn else self.batch_size
        if grad_clip is None and self.is_cnn:
            grad_clip = 10.0

        # Use dueling variants if requested
        if use_dueling:
            if qnetwork_class == QNetwork:
                qnetwork_class = DuelingQNetwork
            elif qnetwork_class == DQNNatureNetwork:
                qnetwork_class = DuelingDQNNatureNetwork
        
        self.replay_start_size = replay_start_size
        self.grad_clip = grad_clip
        self.use_prioritized = use_prioritized
        self.use_double_dqn = use_double_dqn
        self.use_noisy_nets = use_noisy_nets
        self.use_dueling = use_dueling

        # Q-Network
        self.qnetwork_local = qnetwork_class(state_size, action_size, seed, use_noisy_nets=self.use_noisy_nets).to(device)
        self.qnetwork_target = qnetwork_class(state_size, action_size, seed, use_noisy_nets=self.use_noisy_nets).to(device)
        if self.is_cnn:
            self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=lr, alpha=0.95, eps=0.01)
        else:
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory - select between standard and prioritized
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # N-step return tracking
        self.n_step_buffer = deque(maxlen=n_steps)  # Store n-step experiences
        self.gamma_power = GAMMA ** n_steps  # Gamma^n for n-step returns
    
    def step(self, state, action, reward, next_state, done):
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Process n-step experience
        if len(self.n_step_buffer) == self.n_steps or done:
            # Compute n-step return
            n_step_return = 0.0
            for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
                n_step_return += r * (GAMMA ** i)
                if d:  # Episode ends before n steps
                    break
            
            # Get initial state and final next_state
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            last_state, _, _, last_next_state, last_done = self.n_step_buffer[-1]
            
            # Store n-step experience in replay memory
            self.memory.add(first_state, first_action, n_step_return, last_next_state, last_done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= max(self.batch_size, self.replay_start_size):
                sample_result = self.memory.sample()
                if self.use_prioritized:
                    experiences, indices, weights = sample_result
                    self.learn(experiences, self.gamma_power, indices=indices, is_weights=weights)
                else:
                    self.learn(sample_result, self.gamma_power)
            
            # Resample noise for Noisy Networks
            if self.use_noisy_nets:
                self.qnetwork_local.reset_noise()
                self.qnetwork_target.reset_noise()

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

        # Epsilon-greedy action selection (or learned exploration with Noisy Networks)
        if self.use_noisy_nets:
            # With Noisy Networks, noise in weights provides exploration
            # No epsilon-greedy needed; always select greedy action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Standard epsilon-greedy
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, indices=None, is_weights=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor (GAMMA for 1-step, GAMMA^n for n-step returns)
            indices (np.ndarray): indices in replay buffer (for prioritized replay)
            is_weights (torch.Tensor): importance-sampling weights (for prioritized replay)
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states)
        if self.use_double_dqn:
            # Double DQN: use local network to select action, target network to evaluate
            next_action_local = self.qnetwork_local(next_states).detach().argmax(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_action_local)
        else:
            # Standard DQN: use target network for both selection and evaluation
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute TD-error (for priority updates in PER)
        td_error = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()

        # Compute loss
        if self.is_cnn:
            loss = F.smooth_l1_loss(Q_expected, Q_targets, reduction='none')
        else:
            loss = F.mse_loss(Q_expected, Q_targets, reduction='none')
        
        # Apply importance-sampling weights if using prioritized replay
        if is_weights is not None:
            loss = loss * is_weights.unsqueeze(1)
        
        loss = loss.mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip)
        self.optimizer.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized and indices is not None:
            self.memory.update_priorities(indices, td_error.flatten())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        # Resample noise for Noisy Networks after network update
        if self.use_noisy_nets:
            self.qnetwork_local.reset_noise()
            self.qnetwork_target.reset_noise()

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

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer (uniform sampling variant)."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        np.random.seed(seed)
        
        # Use lists to maintain order and allow indexing
        self.memory = []
        self.priorities = np.array([])
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximum priority."""
        e = self.experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
            self.priorities = np.append(self.priorities, self.max_priority)
        else:
            # Overwrite oldest when buffer is full
            idx = len(self.memory) % self.buffer_size
            self.memory[idx] = e
            self.priorities[idx] = self.max_priority
    
    def sample(self):
        """Sample a batch of experiences based on priorities.
        
        Returns:
            experiences: tuple of (states, actions, rewards, next_states, dones)
            indices: indices of sampled experiences (for priority updates)
            is_weights: importance-sampling weights
        """
        # Compute sampling probabilities from priorities
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** PER_ALPHA
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities, replace=False)
        
        # Compute importance-sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-PER_BETA)
        weights /= weights.max()  # Normalize by max for stability
        weights = torch.from_numpy(weights).float().to(device)
        
        # Gather experiences
        experiences = [self.memory[idx] for idx in indices]
        
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-errors.
        
        Params
        ======
            indices: indices of experiences to update
            td_errors: TD-error values for those experiences
        """
        # Clip TD-errors to avoid extreme priorities
        td_errors = np.clip(td_errors, 0, None)
        new_priorities = td_errors + PER_EPSILON
        
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max())
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)