import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# ======================== DQN HYPERPARAMETERS ========================
# These constants define the behavior and learning rate of the DQN algorithm

BUFFER_SIZE = int(1e5)  # Size of the replay buffer - stores up to 100,000 experiences to sample from for training
BATCH_SIZE = 64         # Number of experiences to sample from replay buffer for each training step
GAMMA = 0.99            # Discount factor - determines how much weight is given to future rewards (0.99 = prioritize long-term rewards)
TAU = 1e-3              # Soft update parameter - controls the blend between local and target network (θ_target = τ*θ_local + (1-τ)*θ_target)
LR = 5e-4               # Learning rate for Adam optimizer - controls step size of gradient descent updates
UPDATE_EVERY = 4        # Frequency of network updates - train the network every 4 steps (not every step) to break correlation between samples

# Select device (GPU if available, CPU otherwise) - PyTorch computations will run on this device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment using Deep Q-Network (DQN) algorithm."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed for reproducibility
        """
        # Store state and action dimensions - needed to define neural network architecture
        self.state_size = state_size
        self.action_size = action_size
        
        # Set random seed for reproducibility - ensures same random sequence across runs
        self.seed = random.seed(seed)

        # ======================== Q-NETWORK ARCHITECTURE ========================
        # DQN uses two identical neural networks with the same architecture:
        # 1. Local Network: Updated every step via gradient descent (learning happens here)
        # 2. Target Network: Updates slowly (every TAU fraction) to provide stable Q-value targets
        #    This dual-network approach reduces harmful correlations in the training process
        
        # Initialize local Q-network - this network is trained to predict Q-values for the current state
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        
        # Initialize target Q-network - identical to local but lags behind to provide stable targets
        # The target network provides the Q-value targets when calculating the loss function
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        # Initialize Adam optimizer - optimizes local network weights using gradient descent
        # Only optimizes local_network parameters (not target network)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # ======================== EXPERIENCE REPLAY BUFFER ========================
        # Replay buffer stores recent experiences (state, action, reward, next_state, done)
        # Randomly sampling from this buffer breaks temporal correlations and improves learning stability
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Counter for tracking when to update the network - used to implement UPDATE_EVERY mechanism
        # Training every step leads to instability; waiting UPDATE_EVERY steps helps stabilize learning
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Execute one step of the DQN learning process called after each environment interaction.
        
        This method is the core of the DQN experience replay mechanism:
        1. Stores the current experience in the replay buffer
        2. Periodically samples from the buffer and trains the network
        
        This is called once per environment step with the experience from that step.
        """
        # ======================== STORE EXPERIENCE IN REPLAY BUFFER ========================
        # Add the current experience (s, a, r, s', done) to the replay buffer
        # This enables experience replay: sampling random batches to break temporal correlations
        self.memory.add(state, action, reward, next_state, done)
        
        # ======================== LEARN FROM BATCH (Every UPDATE_EVERY Steps) ========================
        # Increment time step counter and wrap around using modulo operation
        # This creates a cycle: 0, 1, 2, 3, 0, 1, 2, 3, ... with period UPDATE_EVERY=4
        # Only trains when t_step reaches 0 (every UPDATE_EVERY time steps)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        # Check if it's time to train the network (every UPDATE_EVERY steps)
        if self.t_step == 0:
            # Only train if we have enough experiences in the buffer for a full batch
            # This prevents training with too few samples, ensuring statistical validity
            if len(self.memory) > BATCH_SIZE:
                # Sample a random batch of experiences from the replay buffer
                # Random sampling breaks temporal correlations in the data
                experiences = self.memory.sample()
                
                # Train the local Q-network using the sampled batch and the discount factor
                # This performs one gradient descent update on the local network
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Select an action using epsilon-greedy policy based on current Q-values.
        
        This method balances exploration vs exploitation:
        - With probability (1-eps): Select action with highest Q-value (exploitation)
        - With probability eps: Select random action (exploration)
        
        Params
        ======
            state (array_like): current state from environment
            eps (float): epsilon value for epsilon-greedy action selection (0.0 to 1.0)
        """
        # ======================== PREPARE STATE FOR NETWORK ========================
        # Convert state from numpy array to PyTorch tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # unsqueeze(0) adds batch dimension: shape (state_size,) -> (1, state_size)
        # to(device) moves tensor to GPU or CPU based on device variable
        
        # ======================== FORWARD PASS THROUGH LOCAL NETWORK ========================
        # Set network to evaluation mode (disables dropout/batch normalization updates)
        self.qnetwork_local.eval()
        
        # Compute Q-values for all actions without calculating gradients
        # with torch.no_grad() disables gradient calculation for efficiency (not training here)
        with torch.no_grad():
            # Forward pass: network outputs Q-values for each possible action
            # Output shape: (1, action_size) where each value is the predicted Q-value for that action
            action_values = self.qnetwork_local(state)
        
        # Set network back to training mode (re-enables dropout/batch norm for next training step)
        self.qnetwork_local.train()

        # ======================== EPSILON-GREEDY ACTION SELECTION ========================
        # If random value is greater than epsilon: exploit (select best action)
        if random.random() > eps:
            # Select action with highest Q-value (greedy selection)
            # .cpu() moves tensor back to CPU for numpy conversion if it was on GPU
            # .data.numpy() converts PyTorch tensor to numpy array
            # np.argmax finds the index of the maximum Q-value
            # This is the greedy action: the action the network thinks is best
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # With probability eps: explore by selecting random action
            # np.arange(self.action_size) creates array [0, 1, 2, ..., action_size-1]
            # random.choice randomly selects one action from this array
            # This exploration helps the agent discover better strategies it might have missed
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update the local Q-network using the DQN learning algorithm.
        
        This method implements the core of the DQN algorithm:
        1. Compute target Q-values using the target network
        2. Compute predicted Q-values using the local network
        3. Calculate loss between predicted and target Q-values
        4. Update local network weights via backpropagation
        5. Perform soft update on the target network

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tensors sampled from replay buffer
            gamma (float): discount factor for future rewards (typically 0.99)
        """
        # ======================== UNPACK BATCH ========================
        # Unpack the experience tuple into separate tensors
        # Each tensor has shape (BATCH_SIZE, ...) and contains data for a batch of experiences
        states, actions, rewards, next_states, dones = experiences

        # ======================== COMPUTE TARGET Q-VALUES ========================
        # This is the critical DQN Bellman equation: Q(s,a) = r + γ * max_a' Q(s', a')
        # Use target network (not local) to provide stable, slowly-changing targets
        
        # Forward pass through target network: predict Q-values for next states
        # Output shape: (BATCH_SIZE, action_size) with Q-value for each action
        Q_targets_next = self.qnetwork_target(next_states)
        
        # .detach() removes gradients - we don't want to backprop through the target network
        # Target network is only updated via soft_update, not via gradient descent
        Q_targets_next = Q_targets_next.detach()
        
        # Select the maximum Q-value for each next state (best action in that state)
        # .max(1) returns max along dimension 1 (action dimension)
        # [0] extracts the max values (not the indices)
        # Shape after .max(1)[0]: (BATCH_SIZE,) - one value per state in the batch
        Q_targets_next = Q_targets_next.max(1)[0]
        
        # Add back a dimension to match shape: (BATCH_SIZE,) -> (BATCH_SIZE, 1)
        # This is needed for proper broadcasting with rewards and dones
        Q_targets_next = Q_targets_next.unsqueeze(1)
        
        # Compute the actual target Q-values using the Bellman equation:
        # Q_target = reward + gamma * max_Q(next_state) * (1 - done)
        # The (1 - done) term is crucial: if episode is done, there's no future reward (next state has no value)
        # Shape: (BATCH_SIZE, 1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # ======================== COMPUTE PREDICTED Q-VALUES ========================
        # Forward pass through local (learning) network to get Q-values for current states
        # Output shape: (BATCH_SIZE, action_size)
        Q_expected = self.qnetwork_local(states)
        
        # Use .gather(1, actions) to select only the Q-values for the actions that were actually taken
        # .gather(1, actions) selects along dimension 1 (actions) using indices in 'actions' tensor
        # This extracts only the Q-values corresponding to the actions taken (not all actions)
        # Shape after gather: (BATCH_SIZE, 1) - one Q-value per experience (the value of the action taken)
        Q_expected = Q_expected.gather(1, actions)

        # ======================== COMPUTE LOSS AND UPDATE WEIGHTS ========================
        # Calculate Mean Squared Error (MSE) loss between predicted and target Q-values
        # This measures how far off our predictions are from the target values
        # MSE is commonly used in DQN: loss = mean((Q_expected - Q_target)^2)
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Zero out gradients from previous backward pass
        # PyTorch accumulates gradients by default, so we must clear them before computing new gradients
        self.optimizer.zero_grad()
        
        # Backpropagation: compute gradients of loss with respect to all network parameters
        # This calculates dLoss/dWeight for every weight in the local network
        loss.backward()
        
        # Perform one gradient descent step on the local network
        # Updates network weights using Adam optimizer: weights -= lr * gradient
        # Only the local network is updated; target network is updated separately via soft_update
        self.optimizer.step()

        # ======================== SOFT UPDATE TARGET NETWORK ========================
        # Gradually update the target network to track the local network
        # This prevents the target network from changing too abruptly, which would destabilize learning
        # Formula: θ_target = TAU * θ_local + (1 - TAU) * θ_target
        # With TAU=1e-3, the target network moves only 0.1% toward the local network each update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Perform soft update of target network parameters.
        
        Soft update gradually blends local network weights into the target network.
        This prevents the target network from changing too abruptly, which would cause
        training instability. The target network provides stable Q-value targets for learning.
        
        Update rule: θ_target = τ*θ_local + (1 - τ)*θ_target
        With τ=1e-3: target network moves only 0.1% toward local network per update.
        This means it takes many updates for the target network to track the local network.

        Params
        ======
            local_model (PyTorch model): source model - the network being trained
            target_model (PyTorch model): destination model - the slowly-updating network
            tau (float): soft update parameter (typically 1e-3), controls update rate
                        - tau=1.0: hard update (target = local)
                        - tau=0.0: no update (target unchanged)
                        - tau=1e-3: slow soft update (used in DQN)
        """
        # Iterate through all parameters in both models simultaneously
        # zip pairs up corresponding parameters from both networks
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Apply the soft update equation to each parameter:
            # target_param = tau * local_param + (1 - tau) * target_param
            # 
            # This weighted average keeps most of the old target weights (1-tau = 99.9%)
            # while incorporating a small amount of the new local weights (tau = 0.1%)
            # 
            # .data allows direct modification of the tensor values without tracking gradients
            # .copy_() performs the update in-place on the target parameter
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store and replay experience tuples.
    
    Experience replay is a key component of DQN that:
    1. Stores past experiences (state, action, reward, next_state, done)
    2. Breaks temporal correlations by sampling random batches during training
    3. Improves data efficiency by reusing experiences multiple times
    4. Stabilizes learning by using diverse samples from different time periods
    
    The buffer has a fixed maximum size (BUFFER_SIZE). When full, adding new experiences
    overwrites the oldest ones, creating a rolling window of recent experiences.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action (needed for Experience namedtuple)
            buffer_size (int): maximum size of buffer (e.g., 100,000 experiences)
            batch_size (int): size of each training batch sampled from buffer
            seed (int): random seed for reproducibility of sampling
        """
        # Store action size for reference when creating experiences
        self.action_size = action_size
        
        # Create deque (double-ended queue) with maximum length = buffer_size
        # deque with maxlen automatically removes oldest element when exceeding capacity
        # This maintains a rolling window of the most recent experiences
        # Shape: stores up to buffer_size experiences
        self.memory = deque(maxlen=buffer_size)
        
        # Store batch size - determines how many experiences to sample in each batch
        self.batch_size = batch_size
        
        # Create a namedtuple to structure experience data
        # namedtuple provides a clean way to access experience components by name
        # Instead of accessing by index like tuple[0], we can use tuple.state, tuple.action, etc.
        # Fields: state (current observation), action (action taken), 
        #         reward (immediate reward), next_state (resulting observation), done (terminal flag)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        # Set random seed for reproducibility - ensures same random sequence across runs
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the replay buffer.
        
        This method is called after each environment step to store the transition
        (s, a, r, s', done) for later training. Transitions are sampled randomly
        during training to break temporal correlations.

        Params
        ======
            state: current state observed from environment
            action: action taken by the agent
            reward: immediate reward received from environment
            next_state: state resulting from taking the action
            done: boolean flag indicating if the episode terminated
        """
        # Create an Experience namedtuple containing all 5 components of the transition
        # This packages all information about one interaction into a single object
        e = self.experience(state, action, reward, next_state, done)
        
        # Append the experience to the deque (replay buffer memory)
        # If the buffer is full, the oldest experience is automatically removed
        # This creates a sliding window of recent experiences
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from the replay buffer.
        
        Returns experiences as PyTorch tensors ready for neural network training.
        Sampling is random (not sequential) to break temporal correlations in the data.
        Without experience replay, consecutive samples would be highly correlated,
        leading to poor learning and training instability.
        
        Returns: tuple of (states, actions, rewards, next_states, dones) - all are PyTorch tensors
        """
        # Randomly sample batch_size experiences from the memory buffer
        # random.sample(population, k) returns k unique random elements from population
        # Sampling without replacement ensures no duplicate experiences in one batch
        # Each sample is an Experience namedtuple containing (state, action, reward, next_state, done)
        experiences = random.sample(self.memory, k=self.batch_size)

        # ======================== CONVERT STATES TO TENSOR ========================
        # Extract state from each experience in the batch
        # [e.state for e in experiences if e is not None] creates list of states
        # if e is not None check handles potential None values (defensive programming)
        # np.vstack stacks states vertically into 2D array: shape (BATCH_SIZE, state_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # torch.from_numpy converts numpy array to PyTorch tensor
        # .float() converts to float32 type (standard for neural networks)
        # .to(device) moves tensor to GPU or CPU based on device variable
        
        # ======================== CONVERT ACTIONS TO TENSOR ========================
        # Extract action from each experience in the batch
        # Actions are indices (0, 1, 2, ...) so they need to be long (integer) type
        # np.vstack stacks actions into 2D array: shape (BATCH_SIZE, 1)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # .long() converts to int64 type (required for gather operation in learn method)
        
        # ======================== CONVERT REWARDS TO TENSOR ========================
        # Extract reward from each experience in the batch
        # Rewards are typically scalar values (single float per experience)
        # np.vstack stacks rewards into 2D array: shape (BATCH_SIZE, 1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # Float type matches state type for arithmetic operations in learn method
        
        # ======================== CONVERT NEXT STATES TO TENSOR ========================
        # Extract next_state from each experience in the batch
        # Next states have same shape as states
        # np.vstack stacks next_states into 2D array: shape (BATCH_SIZE, state_size)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # Same conversion as states: numpy -> tensor, float32, move to device
        
        # ======================== CONVERT DONE FLAGS TO TENSOR ========================
        # Extract done flag from each experience in the batch
        # done flags are booleans (True if episode ended, False otherwise)
        # np.vstack stacks done flags into 2D array: shape (BATCH_SIZE, 1)
        # .astype(np.uint8) converts booleans to 0/1 integers (PyTorch prefers numeric types)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # .float() converts 0/1 integers to floats, then to(device) moves to proper device
        # Used in learn method as (1 - dones) to zero out Q-values for terminal states
  
        # Return all tensors as a tuple matching the order expected by learn() method
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of the replay buffer (number of stored experiences).
        
        Returns: integer - number of experiences currently in memory
                         (0 to buffer_size, up to maximum capacity)
        
        This is used to check if we have enough experiences to sample a full batch.
        Training only begins when len(memory) >= BATCH_SIZE.
        """
        # Return the current length of the deque
        # len(self.memory) is O(1) operation - very fast
        # This tells us if we've collected enough experiences for a full batch
        return len(self.memory)