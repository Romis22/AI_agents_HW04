import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """Deep Q-Network pro Sportku"""
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()  # Sigmoid pro pravděpodobnosti
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN agent pro hraní Sportky"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Aktualizuje target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Uloží zkušenost do paměti"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Vybere akci pomocí epsilon-greedy strategie"""
        if training and np.random.random() <= self.epsilon:
            # Random action - vyber 6 náhodných čísel + náhodná Šance
            action = np.zeros(50)
            # Vyber 6 náhodných pozic pro čísla
            selected_positions = np.random.choice(49, 6, replace=False)
            action[selected_positions] = 1.0
            # Náhodná Šance
            action[49] = np.random.random()
            return action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.cpu().data.numpy()[0]
    
    def replay(self):
        """Trénuje model na batch ze zkušeností"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        # Vypočítej target Q-values
        target_q_values = current_q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i] = torch.FloatTensor(actions[i]) * rewards[i]
            else:
                # Pro kontinuální akční prostor použij MSE loss
                target_q_values[i] = torch.FloatTensor(actions[i]) * (rewards[i] + 0.95 * torch.max(next_q_values[i]))
        
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Uloží model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model uložen do {filepath}")
    
    def load(self, filepath):
        """Načte model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model načten z {filepath}")

class RandomAgent:
    """Náhodný agent pro porovnání"""
    
    def __init__(self):
        pass
    
    def act(self, state=None):
        """Vybere náhodnou akci"""
        action = np.zeros(50)
        # Vyber 6 náhodných pozic pro čísla
        selected_positions = np.random.choice(49, 6, replace=False)
        action[selected_positions] = 1.0
        # Náhodná Šance (50% pravděpodobnost jako pravděpodobnostní hodnota)
        action[49] = np.random.random()
        return action