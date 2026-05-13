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
from collections import deque

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
learning_rate = 1e-3
epochs = 1000
batch_size = 32
max_steps = 50

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# Model Definition
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

def train_naive_dqn():
    print("Training Naive DQN...")
    model = QNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    eps = epsilon
    
    for epoch in range(epochs):
        game = Gridworld(size=4, mode='static')
        state = preprocess_state(game.board.render_np())
        status = 0
        step = 0
        
        while status == 0 and step < max_steps:
            q_values = model(state)
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                action = torch.argmax(q_values).item()
                
            game.makeMove(action_set[action])
            reward = game.reward()
            
            next_state = preprocess_state(game.board.render_np())
            status = check_win_loss(game)
            
            with torch.no_grad():
                next_q_values = model(next_state)
                max_next_q = torch.max(next_q_values).item()
                
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
            
        if eps > epsilon_min:
            eps -= (1.0 / epochs)
            
    return losses

def train_er_dqn():
    print("Training DQN with Experience Replay...")
    model = QNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=1000)
    
    losses = []
    eps = epsilon
    
    for epoch in range(epochs):
        game = Gridworld(size=4, mode='static')
        state = preprocess_state(game.board.render_np())
        status = 0
        step = 0
        
        while status == 0 and step < max_steps:
            q_values = model(state)
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                action = torch.argmax(q_values).item()
                
            game.makeMove(action_set[action])
            reward = game.reward()
            
            next_state = preprocess_state(game.board.render_np())
            status = check_win_loss(game)
            
            replay_buffer.append((state, action, reward, next_state, status))
            
            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                
                state_batch = torch.cat([m[0] for m in minibatch])
                action_batch = torch.tensor([m[1] for m in minibatch]).to(device)
                reward_batch = torch.tensor([m[2] for m in minibatch], dtype=torch.float32).to(device)
                next_state_batch = torch.cat([m[3] for m in minibatch])
                status_batch = torch.tensor([m[4] for m in minibatch], dtype=torch.float32).to(device)
                
                q_batch = model(state_batch)
                with torch.no_grad():
                    next_q_batch = model(next_state_batch)
                    max_next_q_batch = torch.max(next_q_batch, dim=1)[0]
                    
                target_batch = reward_batch + gamma * max_next_q_batch * (1 - torch.abs(status_batch))
                
                target_q_batch = q_batch.clone()
                for i in range(batch_size):
                    target_q_batch[i][action_batch[i]] = target_batch[i]
                
                loss = criterion(q_batch, target_q_batch.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
            state = next_state
            step += 1
            
        if eps > epsilon_min:
            eps -= (1.0 / epochs)
            
    return losses

if __name__ == "__main__":
    # 1. Train and Plot Naive DQN
    naive_loss = train_naive_dqn()
    plt.figure(figsize=(10, 7))
    plt.plot(naive_loss, color='blue')
    plt.xlabel("Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Naive DQN Training Loss (Static Mode)", fontsize=13)
    plt.savefig('hw3_1_naive_results.png')
    print("Saved plot to hw3_1_naive_results.png")
    plt.close()
    
    # 2. Train and Plot ER DQN
    er_loss = train_er_dqn()
    plt.figure(figsize=(10, 7))
    plt.plot(er_loss, color='green')
    plt.xlabel("Steps", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("DQN with Experience Replay Training Loss (Static Mode)", fontsize=13)
    plt.savefig('hw3_1_er_results.png')
    print("Saved plot to hw3_1_er_results.png")
    plt.close()
