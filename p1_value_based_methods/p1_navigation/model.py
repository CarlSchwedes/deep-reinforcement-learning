import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNNatureNetwork(nn.Module):
    """Convolutional Q-Network (Nature DQN paper).
    
    Processes a sequence of 4 stacked grayscale frames (84x84x4) through convolutional layers.
    Unlike typical image channels, the 4 "channels" each represent a consecutive frame in time,
    allowing the network to capture motion, velocity, and temporal dynamics.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (tuple): Shape of state (frames, height, width) - expect (4, 84, 84)
                               where 4 is the number of consecutive frames, not color channels
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in hidden fully-connected layer
        """
        super(DQNNatureNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
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
