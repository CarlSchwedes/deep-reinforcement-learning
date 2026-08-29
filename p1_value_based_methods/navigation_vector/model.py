import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model for Dueling DQN."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model."""
        super(DuelingQNetwork, self).__init__()
        
        # Feature extraction layer shared by both streams
        self.feature_layer = nn.Linear(state_size, fc1_units)
        
        # State Value Stream V(s) - Outputs 1 single scalar value for the state
        self.value_fc = nn.Linear(fc1_units, fc2_units)
        self.value_out = nn.Linear(fc2_units, 1)
        
        # Action Advantage Stream A(s, a) - Outputs an advantage value for each action
        self.advantage_fc = nn.Linear(fc1_units, fc2_units)
        self.advantage_out = nn.Linear(fc2_units, action_size)

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
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values
