import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """Noisy Linear layer for learning exploration (Noisy Networks).
    
    Implements learnable parameter noise following Fortunato et al., 2018.
    The noise is sampled from a factorized Gaussian distribution and resampled at each forward pass.
    """
    
    def __init__(self, in_features, out_features, std_init=0.5):
        """Initialize a Noisy Linear layer.
        
        Params
        ======
            in_features (int): Size of input features
            out_features (int): Size of output features
            std_init (float): Initial standard deviation of weight noise
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # Learnable noise standard deviations
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (non-learnable, resampled each forward pass)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with appropriate distributions."""
        mu_range = 1.0 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_sigma.data.fill_(self.std_init / 1.0)
    
    def _sample_noise(self):
        """Sample noise from factorized Gaussian distribution."""
        # Sample input and output noise
        epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device)
        epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device)
        
        # Transform to factorized form (reduces correlation)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, input):
        """Forward pass with learned parameter noise.
        
        Params
        ======
            input (torch.Tensor): Input tensor
        
        Returns
        ======
            Output with noisy weights and biases applied
        """
        if self.training:
            # During training, sample noise and add to weights/biases
            self._sample_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only the mean (deterministic)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            use_noisy_nets (bool): Whether to use Noisy Networks for learned exploration
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_noisy_nets = use_noisy_nets
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # Use Noisy Linear layer for output if enabled
        if use_noisy_nets:
            self.fc3 = NoisyLinear(fc2_units, action_size)
        else:
            self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def reset_noise(self):
        """Resample noise in all Noisy Layers (called before each episode)."""
        if self.use_noisy_nets and hasattr(self.fc3, '_sample_noise'):
            self.fc3._sample_noise()


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network for vector observations.
    
    Separates value and advantage streams for improved representation learning.
    Q(s, a) = V(s) + [A(s, a) - mean(A(s, a))]
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            use_noisy_nets (bool): Whether to use Noisy Networks for learned exploration
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_noisy_nets = use_noisy_nets
        self.action_size = action_size
        
        # Shared feature layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        # Value stream
        self.fc3_value = nn.Linear(fc2_units, 32)
        if use_noisy_nets:
            self.value = NoisyLinear(32, 1)
        else:
            self.value = nn.Linear(32, 1)
        
        # Advantage stream
        self.fc3_adv = nn.Linear(fc2_units, 32)
        if use_noisy_nets:
            self.advantage = NoisyLinear(32, action_size)
        else:
            self.advantage = nn.Linear(32, action_size)

    def forward(self, state):
        """Compute Q-values by combining value and advantage streams."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Value stream
        v = F.relu(self.fc3_value(x))
        v = self.value(v)
        
        # Advantage stream
        a = F.relu(self.fc3_adv(x))
        a = self.advantage(a)
        
        # Combine: Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
    
    def reset_noise(self):
        """Resample noise in all Noisy Layers."""
        if self.use_noisy_nets:
            if hasattr(self.value, '_sample_noise'):
                self.value._sample_noise()
            if hasattr(self.advantage, '_sample_noise'):
                self.advantage._sample_noise()


class DQNNatureNetwork(nn.Module):
    """Convolutional Q-Network (Nature DQN paper).
    
    Processes a sequence of 4 stacked grayscale frames (84x84x4) through convolutional layers.
    Unlike typical image channels, the 4 "channels" each represent a consecutive frame in time,
    allowing the network to capture motion, velocity, and temporal dynamics.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=512, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state (frames, height, width) - expect (4, 84, 84)
                               where 4 is the number of consecutive frames, not color channels
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in hidden fully-connected layer
            use_noisy_nets (bool): Whether to use Noisy Networks for learned exploration
        """
        super(DQNNatureNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
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


class DuelingDQNNatureNetwork(nn.Module):
    """Dueling CNN Q-Network (Nature DQN + Dueling Architecture).
    
    Processes 4 stacked grayscale frames through convolutional layers with separate
    value and advantage streams for improved representation learning.
    Q(s, a) = V(s) + [A(s, a) - mean(A(s, a))]
    """

    def __init__(self, state_size, action_size, seed, fc1_units=512, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state (frames, height, width) - expect (4, 84, 84)
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in hidden fully-connected layer
            use_noisy_nets (bool): Whether to use Noisy Networks for learned exploration
        """
        super(DuelingDQNNatureNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_noisy_nets = use_noisy_nets
        self.action_size = action_size
        
        # Convolutional layers (Nature DQN architecture)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        
        # Shared FC layer
        # Flattened size: 7 * 7 * 64 = 3136
        self.fc1 = nn.Linear(7 * 7 * 64, fc1_units)
        
        # Value stream
        self.fc_value = nn.Linear(fc1_units, 32)
        if use_noisy_nets:
            self.value = NoisyLinear(32, 1)
        else:
            self.value = nn.Linear(32, 1)
        
        # Advantage stream
        self.fc_adv = nn.Linear(fc1_units, 32)
        if use_noisy_nets:
            self.advantage = NoisyLinear(32, action_size)
        else:
            self.advantage = nn.Linear(32, action_size)

    def forward(self, state):
        """Compute Q-values by combining value and advantage streams.
        
        Params
        ======
            state (torch.Tensor): Frame sequence tensor of shape (batch_size, 4, 84, 84)
        
        Returns
        ======
            Q-values for each action
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        
        # Value stream
        v = F.relu(self.fc_value(x))
        v = self.value(v)
        
        # Advantage stream
        a = F.relu(self.fc_adv(x))
        a = self.advantage(a)
        
        # Combine: Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
    
    def reset_noise(self):
        """Resample noise in all Noisy Layers."""
        if self.use_noisy_nets:
            if hasattr(self.value, '_sample_noise'):
                self.value._sample_noise()
            if hasattr(self.advantage, '_sample_noise'):
                self.advantage._sample_noise()
