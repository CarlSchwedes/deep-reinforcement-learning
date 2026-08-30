import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, use_noisy_nets=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            use_noisy_nets (bool): Whether to use Noisy Networks for exploration
        """
        super(QNetwork, self).__init__()

        self.use_noisy_nets = use_noisy_nets

        if not self.use_noisy_nets:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
        else:
            self.fc1 = NoisyLinear(state_size, fc1_units)
            self.fc2 = NoisyLinear(fc1_units, fc2_units)
            self.fc3 = NoisyLinear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model for Dueling DQN."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, use_noisy_nets=False):
        """Initialize parameters and build model."""
        super(DuelingQNetwork, self).__init__()

        self.use_noisy_nets = use_noisy_nets
    
        # Feature extraction layer shared by both streams
        self.feature_layer = nn.Linear(state_size, fc1_units)
        
        # State Value Stream V(s) - Outputs 1 single scalar value for the state
        if not self.use_noisy_nets:
            self.value_fc = nn.Linear(fc1_units, fc2_units)
            self.value_out = nn.Linear(fc2_units, 1)
        else:
            self.value_fc = NoisyLinear(fc1_units, fc2_units)
            self.value_out = NoisyLinear(fc2_units, 1)
        
        # Action Advantage Stream A(s, a) - Outputs an advantage value for each action
        if not self.use_noisy_nets:
            self.advantage_fc = nn.Linear(fc1_units, fc2_units)
            self.advantage_out = nn.Linear(fc2_units, action_size)
        else:
            self.advantage_fc = NoisyLinear(fc1_units, fc2_units)
            self.advantage_out = NoisyLinear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Extract features
        x = F.relu(self.feature_layer(state))
        
        # Process through Value stream
        v = F.relu(self.value_fc(x))
        value = self.value_out(v)
        
        # Process through Advantage stream
        a = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(a)
        
        # Combine using the stable dueling formula: Q = V + (A - mean(A))
        if not self.use_noisy_nets:
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True)[0])
        else:
            q_values = value + (advantage - advantage.max(dim=-1, keepdim=True)[0])
        
        return q_values

    def reset_noise(self):
        """Triggers a noise recalculation pass through all noisy sub-layers."""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()


class DistributionalDuelingNoisyQNetwork(nn.Module):
    """Full Rainbow Head: Distributional + Dueling + Noisy Network."""

    def __init__(self, state_size, action_size, num_atoms=51, fc1_units=64, fc2_units=64):
        super(DistributionalDuelingNoisyQNetwork, self).__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        
        # Shared input feature layer
        self.feature_layer = nn.Linear(state_size, fc1_units)
        
        # Dueling Streams with NoisyLinear - OUTPUTS EXPANDED TO ATOMS:
        # Value stream outputs 51 atoms (representing the baseline state distribution)
        self.value_fc = NoisyLinear(fc1_units, fc2_units)
        self.value_out = NoisyLinear(fc2_units, num_atoms)
        
        # Advantage stream outputs action_size * num_atoms (4 * 51 atoms)
        self.advantage_fc = NoisyLinear(fc1_units, fc2_units)
        self.advantage_out = NoisyLinear(fc2_units, action_size * num_atoms)

    def forward(self, state):
        x = F.relu(self.feature_layer(state))
        
        # Calculate streams and reshape into tensor structures
        v = F.relu(self.value_fc(x))
        value = self.value_out(v).view(-1, 1, self.num_atoms) # Shape: (Batch, 1, 51)
        
        a = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(a).view(-1, self.action_size, self.num_atoms) # Shape: (Batch, 4, 51)
        
        # Combine value and advantage streams on an atom-by-atom level (max aggregation)
        q_dist = value + (advantage - advantage.mean(dim=1, keepdim=True)) # Shape: (Batch, 4, 51)
        
        # Convert raw logits into clean probabilities across the atom axis using Softmax
        probs = F.softmax(q_dist, dim=-1) 
        return probs # Returns distribution matrices: (Batch, Action_Size, Atoms)

    def reset_noise(self):
        """Reset noise buffers inside all parameter layers."""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
