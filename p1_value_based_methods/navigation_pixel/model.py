import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear layer for systematic parametric exploration."""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters (Weights and Biases)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Fixed buffers for the structural random noise vectors
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize mu and sigma parameters according to the factorized scheme."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_val = self.std_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_val)
        self.bias_sigma.data.fill_(sigma_val)

    def _scale_noise(self, size):
        """Helper function to generate and scale factorized noise vectors."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """Generates new random noise tensors for a forward evaluation pass."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Factorized matrix multiplication for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        """Performs a noisy linear forward pass."""
        if self.training:
            # Add dynamic noise to parameters during the training phase
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use completely clean deterministic parameters during final evaluation testing
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)


class QNetwork(nn.Module):
    """Convolutional Q-Network (Nature DQN paper).
    
    Processes a sequence of 4 stacked grayscale frames (84x84x4) through convolutional layers.
    Unlike typical image channels, the 4 "channels" each represent a consecutive frame in time,
    allowing the network to capture motion, velocity, and temporal dynamics.
    """

    def __init__(self, state_size, action_size, fc1_units=512, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state (frames, height, width) - expect (4, 84, 84)
                               where 4 is the number of consecutive frames, not color channels
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in hidden fully-connected layer
            use_noisy_nets (bool): Whether to use Noisy Networks for learned exploration
        """
        super(QNetwork, self).__init__()
        self.use_noisy_nets = use_noisy_nets
        
        # Convolutional layers (Nature DQN architecture)
        # Input: 4x84x84 (4 consecutive grayscale frames stacked as "channels")
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        # Output: 20x20x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        # Output: 9x9x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        # Output: 7x7x64
        
        # Fully connected layers
        # Flattened size: 7 * 7 * 64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, fc1_units)
        
        # Use Noisy Linear layer for output if enabled
        if use_noisy_nets:
            self.fc2 = NoisyLinear(fc1_units, action_size)
        else:
            self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
        
        Params
        ======
            state (torch.Tensor): Frame sequence tensor of shape (batch_size, 4, 84, 84)
                                   The 4 "channels" are 4 consecutive grayscale frames in time,
                                   allowing the network to capture temporal dynamics and motion.
        
        Returns
        ======
            Q-values for each action
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def reset_noise(self):
        """Resample noise in all Noisy Layers (called before each episode)."""
        if self.use_noisy_nets and hasattr(self.fc2, '_sample_noise'):
            self.fc2._sample_noise()


class DuelingQNetwork(nn.Module):
    """Dueling CNN Q-Network (Nature DQN + Dueling Architecture).
    
    Processes 4 stacked grayscale frames through convolutional layers with separate
    value and advantage streams for improved representation learning.
    Q(s, a) = V(s) + [A(s, a) - mean(A(s, a))]
    """

    def __init__(self, state_size, action_size, fc1_units=512, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state (frames, height, width) - expect (4, 84, 84)
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in hidden fully-connected layer
            use_noisy_nets (bool): Whether to use Noisy Networks for learned exploration
        """
        super(DuelingQNetwork, self).__init__()
        self.action_size = action_size
        self.use_noisy_nets = use_noisy_nets
        
        # Shared Convolutional Feature Layers (Nature DQN Standard)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Flattened size after conv layers: 7 * 7 * 64 = 3136
        flattened_size = 3136
        
        # Helper function to dynamically swap between standard Linear and NoisyLinear layers
        def LinearLayer(in_dim, out_dim):
            return NoisyLinear(in_dim, out_dim) if self.use_noisy_nets else nn.Linear(in_dim, out_dim)
        
        # State-Value Stream V(s) -> Outputs 1 single scalar value for the state
        self.value_fc = LinearLayer(flattened_size, fc1_units)
        self.value_out = LinearLayer(fc1_units, 1)
        
        # Action-Advantage Stream A(s,a) -> Outputs an advantage value for each action
        self.advantage_fc = LinearLayer(flattened_size, fc1_units)
        self.advantage_out = LinearLayer(fc1_units, action_size)

    def forward(self, state):
        # Extract features through standard CNN layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten into a vector: (Batch_Size, 3136)
        x = x.view(x.size(0), -1)
        
        # Process through Value stream
        v = F.relu(self.value_fc(x))
        value = self.value_out(v) # Shape: (Batch_Size, 1)
        
        # Process through Advantage stream
        a = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(a) # Shape: (Batch_Size, Action_Size)
        
        # Combine using the standard max aggregation formula for scalar Q-values
        q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        
        return q_values # Shape: (Batch_Size, Action_Size)

    def reset_noise(self):
        """Reset noise buffers inside all parameter layers if NoisyNet is enabled."""
        if self.use_noisy_nets:
            self.value_fc.reset_noise()
            self.value_out.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage_out.reset_noise()


class DistributionalDuelingQNetwork(nn.Module):
    """Full Rainbow Convolutional Head: Distributional + Dueling + Optional Noisy Network."""

    def __init__(self, state_size, action_size, num_atoms=51, fc1_units=512, use_noisy_nets=False):
        super(DistributionalDuelingQNetwork, self).__init__() # Modern, clean Python 3 initialization
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.use_noisy_nets = use_noisy_nets
        
        # Shared Convolutional Feature Layers (Nature DQN Standard)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Flattened size after conv layers: 7 * 7 * 64 = 3136
        flattened_size = 3136
        
        # Helper function to dynamically swap between standard Linear and NoisyLinear layers
        def LinearLayer(in_dim, out_dim):
            return NoisyLinear(in_dim, out_dim) if self.use_noisy_nets else nn.Linear(in_dim, out_dim)
        
        # Dueling Streams - OUTPUTS EXPANDED TO ATOMS:
        # Value stream outputs 51 atoms representing the state distribution baseline
        self.value_fc = LinearLayer(flattened_size, fc1_units)
        self.value_out = LinearLayer(fc1_units, num_atoms)
        
        # Advantage stream outputs action_size * num_atoms (4 actions * 51 atoms)
        self.advantage_fc = LinearLayer(flattened_size, fc1_units)
        self.advantage_out = LinearLayer(fc1_units, action_size * num_atoms)

    def forward(self, state):           
        # Extract visual features through CNN layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten into a vector: (Batch_Size, 3136)
        x = x.view(x.size(0), -1)
        
        # Calculate Value stream distribution
        v = F.relu(self.value_fc(x))
        value = self.value_out(v).view(-1, 1, self.num_atoms) # Shape: (Batch, 1, 51)
        
        # Calculate Advantage stream distribution
        a = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(a).view(-1, self.action_size, self.num_atoms) # Shape: (Batch, Action_Size, 51)
        
        # Combine value and advantage streams on an atom-by-atom level (mean aggregation for C51)
        q_dist = value + (advantage - advantage.mean(dim=1, keepdim=True)) # Shape: (Batch, Action_Size, 51)
        
        # Convert raw logits into clean probabilities across the atom axis using Softmax
        probs = F.softmax(q_dist, dim=-1)
        return probs # Returns distribution matrices: (Batch_Size, Action_Size, Atoms)

    def reset_noise(self):
        """Reset noise buffers inside all parameter layers if NoisyNet is enabled."""
        if self.use_noisy_nets:
            self.value_fc.reset_noise()
            self.value_out.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage_out.reset_noise()