import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import torch.distributions as distributions

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.xavier_uniform_(self.log_std_layer.weight)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.q_layer.weight)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        q_value = self.q_layer(x)
        return q_value

class SACAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.00015, lr_critic=0.00015, gamma=0.99, tau=0.005, alpha=0.05):
        super(SACAgent, self).__init__()

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.critic1_target = CriticNetwork(state_dim, action_dim)
        self.critic2_target = CriticNetwork(state_dim, action_dim)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr_critic)

    def act(self, state, explore=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            if explore:
                z = normal.rsample()
            else:
                z = mean
            action = torch.tanh(z)
        self.actor.train()
        return action.cpu().detach().numpy()[0]

    def learn(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        batch = memory.sample(batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        next_state = torch.FloatTensor(np.array(next_state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_state)
            next_std = next_log_std.exp()
            normal = torch.distributions.Normal(next_mean, next_std)
            next_z = normal.rsample()
            next_action = torch.tanh(next_z)
            log_prob = normal.log_prob(next_z) - torch.log(1 - next_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = reward + (1- done) * self.gamma * target_v

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        loss_critic1 = nn.MSELoss()(current_q1, target_q)
        loss_critic2 = nn.MSELoss()(current_q2, target_q)

        self.optimizer_critic1.zero_grad()
        loss_critic1.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        loss_critic2.backward()
        self.optimizer_critic2.step()

        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        new_action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - new_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        critic_value = torch.min(q1_new, q2_new)
        loss_actor = (self.alpha * log_prob - critic_value).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        for target_param, local_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
