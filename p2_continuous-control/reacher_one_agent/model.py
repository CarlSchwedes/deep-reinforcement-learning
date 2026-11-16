import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class DecayedGaussianNoise(nn.Module):
    """Generates zero-mean Gaussian noise with exponentially decaying standard deviation."""

    def __init__(self, action_size, initial_std=0.2, min_std=0.01, decay_rate=0.995):
        super(DecayedGaussianNoise, self).__init__()
        self.action_size = action_size
        self.initial_std = initial_std
        self.min_std = min_std
        self.decay_rate = decay_rate
        self.std = initial_std

    def forward(self, actions):
        """Sample noise matching the shape, device, and dtype of the action tensor."""
        return torch.randn_like(actions) * self.std

    def decay(self):
        """Decay the standard deviation down to min_std."""
        self.std = max(self.min_std, self.std * self.decay_rate)

    def reset(self):
        """Reset standard deviation back to initial value."""
        self.std = self.initial_std


class DeepDeterministicPolicyGradientNetwork(nn.Module):
    """Actor-Critic Network for DDPG."""

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300):
        super(DeepDeterministicPolicyGradientNetwork, self).__init__()

        # Actor Network
        self.actor_fc1 = nn.Linear(state_size, fc1_units)
        self.actor_fc2 = nn.Linear(fc1_units, fc2_units)
        self.actor_out = nn.Linear(fc2_units, action_size)
        
        # Critic Network
        self.critic_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic_out = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Forward pass for the actor network."""
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        return torch.tanh(self.actor_out(x))  # Output actions in range [-1, 1]

    def critic_forward(self, state, action):
        """Forward pass for the critic network."""
        x = torch.cat((state, action), dim=1)  # Concatenate state and action
        x = F.relu(self.critic_fc1(x))
        x = F.relu(self.critic_fc2(x))
        return self.critic_out(x)  # Output Q-value


class DeepDeterministicPolicyGradientNetworkWithTD3AndDecayingNoise(nn.Module):
    """Actor-Critic Network for DDPG with TD3 and decayed Gaussian noise."""

    def __init__(
        self,
        state_size,
        action_size,
        fc1_units=400,
        fc2_units=300,
        initial_std=0.2,
        min_std=0.01,
        decay_rate=0.995,
    ):
        super(DeepDeterministicPolicyGradientNetworkWithTD3AndDecayingNoise, self).__init__()

        # Actor Network
        self.actor_fc1 = nn.Linear(state_size, fc1_units)
        self.actor_fc2 = nn.Linear(fc1_units, fc2_units)
        self.actor_out = nn.Linear(fc2_units, action_size)
        self.exploration_noise = DecayedGaussianNoise(
            action_size=action_size,
            initial_std=initial_std,
            min_std=min_std,
            decay_rate=decay_rate,
        )

        # Critic Networks (Two Q-functions for TD3)
        self.critic1_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic1_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic1_out = nn.Linear(fc2_units, 1)

        self.critic2_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic2_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic2_out = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Forward pass for the actor network."""
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        return torch.tanh(self.actor_out(x))  # Output actions in range [-1, 1]

    @torch.no_grad()
    def act(self, state, add_noise=True):
        """Select bounded actions, adding exploration noise when requested."""
        actions = self.forward(state)
        if add_noise:
            actions = actions + self.exploration_noise(actions)
        return actions.clamp(-1.0, 1.0)

    def decay_noise(self):
        """Reduce the exploration noise standard deviation."""
        self.exploration_noise.decay()

    def reset_noise(self):
        """Reset exploration noise back to its initial standard deviation."""
        self.exploration_noise.reset()

    def critic_forward(self, state, action):
        """Forward pass for the first critic network."""
        x = torch.cat((state, action), dim=1)  # Concatenate state and action
        x = F.relu(self.critic1_fc1(x))
        x = F.relu(self.critic1_fc2(x))
        return self.critic1_out(x)  # Output Q-value from first critic

    def critic2_forward(self, state, action):
        """Forward pass for the second critic network."""
        x = torch.cat((state, action), dim=1)  # Concatenate state and action
        x = F.relu(self.critic2_fc1(x))
        x = F.relu(self.critic2_fc2(x))
        return self.critic2_out(x)  # Output Q-value from second critic


class SoftActorCriticNetwork(nn.Module):
    """Actor-Critic Network for Soft Actor-Critic (SAC)."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256, log_std_min=-20, log_std_max=2):
        super(SoftActorCriticNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Actor Network
        self.actor_fc1 = nn.Linear(state_size, fc1_units)
        self.actor_fc2 = nn.Linear(fc1_units, fc2_units)
        self.actor_mean = nn.Linear(fc2_units, action_size)
        self.actor_log_std = nn.Linear(fc2_units, action_size)
        
        # Critic Networks (Two Q-functions for SAC)
        self.critic1_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic1_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic1_out = nn.Linear(fc2_units, 1)

        self.critic2_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.critic2_fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic2_out = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Forward pass for the actor network."""
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        mean = self.actor_mean(x)
        log_std = torch.clamp(self.actor_log_std(x), min=-20, max=2)  # Clamp log_std for numerical stability
        return mean, log_std

    def sample_action(self, state, deterministic=False):
        """Sample bounded action in [-1, 1] using reparameterization trick and compute log_prob."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            z = mean
        else:
            # Reparameterization trick (rsample keeps gradients flowing through sampling)
            z = normal.rsample()

        action = torch.tanh(z)

        # Change-of-variables correction for tanh squashing
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def critic_forward(self, state, action):
        """Forward pass for the first critic network."""
        x = torch.cat((state, action), dim=1)  # Concatenate state and action
        x = F.relu(self.critic1_fc1(x))
        x = F.relu(self.critic1_fc2(x))
        return self.critic1_out(x)  # Output Q-value from first critic

    def critic2_forward(self, state, action):
        """Forward pass for the second critic network."""
        x = torch.cat((state, action), dim=1)  # Concatenate state and action
        x = F.relu(self.critic2_fc1(x))
        x = F.relu(self.critic2_fc2(x))
        return self.critic2_out(x)  # Output Q-value from second critic
