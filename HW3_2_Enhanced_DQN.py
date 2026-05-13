import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from Gridworld import Gridworld
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
learning_rate = 1e-3
epochs = 1500
max_steps = 50
target_update_freq = 100

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# Basic Q-Network (Used for Double DQN)
class QNetwork(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Dueling Q-Network
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        self.fc2 = nn.Linear(150, 100)
        
        # Value Stream
        self.val_fc = nn.Linear(100, 1)
        # Advantage Stream
        self.adv_fc = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        val = self.val_fc(x)
        adv = self.adv_fc(x)
        
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

def preprocess_state(state):
    return torch.from_numpy(state.flatten()).float().unsqueeze(0).to(device)

def check_win_loss(game):
    player = game.board.components['Player'].pos
    goal = game.board.components['Goal'].pos
    pit = game.board.components['Pit'].pos
    if player == goal:
        return 1
    elif player == pit:
        return -1
    return 0

def train_agent(variant='double'):
    print(f"Training {variant} DQN (No Experience Replay) in player mode...")
    
    if variant == 'dueling':
        model = DuelingQNetwork().to(device)
    else:
        model = QNetwork().to(device)
        
    target_model = copy.deepcopy(model)
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    eps = epsilon
    total_steps = 0
    
    for epoch in range(epochs):
        game = Gridworld(size=4, mode='player')
        state = preprocess_state(game.board.render_np())
        status = 0
        step = 0
        epoch_reward = 0
        
        while status == 0 and step < max_steps:
            q_values = model(state)
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                action = torch.argmax(q_values).item()
                
            game.makeMove(action_set[action])
            reward = game.reward()
            epoch_reward += reward
            
            next_state = preprocess_state(game.board.render_np())
            status = check_win_loss(game)
            
            with torch.no_grad():
                if variant == 'double':
                    # Double DQN: Use primary network to select action, target network to evaluate
                    next_q_primary = model(next_state)
                    best_action = torch.argmax(next_q_primary).item()
                    next_q_target = target_model(next_state)
                    max_next_q = next_q_target.squeeze()[best_action]
                else:
                    # Standard Dueling DQN Target calculation
                    next_q_target = target_model(next_state)
                    max_next_q = torch.max(next_q_target).item()
                    
            target = reward + gamma * max_next_q * (1 - abs(status))
            
            target_q_values = q_values.clone()
            target_q_values[0][action] = target
            
            loss = criterion(q_values, target_q_values.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
                
            state = next_state
            step += 1
            total_steps += 1
            
            if total_steps % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
        if eps > epsilon_min:
            eps -= (1.0 / epochs) # Linear decay like in repo
            
    return losses

if __name__ == "__main__":
    double_losses = train_agent('double')
    dueling_losses = train_agent('dueling')

    plt.figure(figsize=(10, 7))
    plt.plot(double_losses, label='Double DQN Loss', alpha=0.8, color='blue')
    plt.plot(dueling_losses, label='Dueling DQN Loss', alpha=0.8, color='orange')
    plt.title('DQN Variants Training Loss', fontsize=13)
    plt.xlabel('Steps', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.legend()
    
    plt.savefig('hw3_2_results.png')
    print("Saved plot to hw3_2_results.png")
