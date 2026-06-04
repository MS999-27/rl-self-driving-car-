import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.mean(x), torch.clamp(self.log_std(x), min=-20, max=2)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = (normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, -log_prob.mean().item()

class SAC:
    def __init__(self, num_inputs):
        self.policy = PolicyNetwork(num_inputs, 1)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.entropy = 0

    def update(self, reward, new_state):
        state_t = torch.Tensor(new_state).float().unsqueeze(0)
        with torch.no_grad():
            action_t, ent = self.policy.sample(state_t)
            self.entropy = ent
        return action_t.item() * 15, self.entropy