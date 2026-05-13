import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import copy
from Gridworld import Gridworld
import random
from collections import deque

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        state_batch = torch.cat([m[0] for m in minibatch])
        action_batch = torch.tensor([m[1] for m in minibatch])
        reward_batch = torch.tensor([m[2] for m in minibatch], dtype=torch.float32)
        next_state_batch = torch.cat([m[3] for m in minibatch])
        done_batch = torch.tensor([m[4] for m in minibatch], dtype=torch.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        
    def __len__(self):
        return len(self.buffer)

class RLDataset(Dataset):
    def __init__(self, buffer, sample_size):
        self.buffer = buffer
        self.sample_size = sample_size
        
    def __len__(self):
        return self.sample_size
        
    def __getitem__(self, idx):
        return 0

# Dueling Q-Network
class DuelingQNet(nn.Module):
    def __init__(self):
        super(DuelingQNet, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        
        self.val_fc = nn.Linear(100, 1)
        self.adv_fc = nn.Linear(100, 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        val = self.val_fc(x)
        adv = self.adv_fc(x)
        
        return val + (adv - adv.mean(dim=1, keepdim=True))

class DQNLightning(pl.LightningModule):
    def __init__(self, batch_size=64, lr=1e-3, gamma=0.9, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.99):
        super(DQNLightning, self).__init__()
        self.save_hyperparameters()
        
        self.model = DuelingQNet()
        self.target_net = copy.deepcopy(self.model)
        self.target_net.eval()
        
        self.buffer = ReplayBuffer(5000)
        self.criterion = nn.MSELoss()
        
        self.epsilon = epsilon_start
        self.game = Gridworld(size=4, mode='random')
        self.state = self._get_state(self.game)
        
        self.episode_rewards = []
        self.train_losses = []
        self.current_epoch_reward = 0
        self.step_count = 0
        self.epoch_steps = 0
        
    def forward(self, x):
        return self.model(x)
        
    def _get_state(self, game):
        return torch.from_numpy(game.board.render_np().flatten()).float().unsqueeze(0)
        
    def _check_win_loss(self, game):
        player = game.board.components['Player'].pos
        goal = game.board.components['Goal'].pos
        pit = game.board.components['Pit'].pos
        if player == goal: return 1
        elif player == pit: return -1
        return 0

    def play_step(self):
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self(self.state.to(self.device))
                action = torch.argmax(q_values).item()
                
        old_pos = self.game.board.components['Player'].pos
        self.game.makeMove(action_set[action])
        new_pos = self.game.board.components['Player'].pos
        
        status = self._check_win_loss(self.game)
        
        # Anti-wall collision reward shaping
        if new_pos == old_pos and status == 0:
            reward = -10 # Hit wall / boundary
        else:
            reward = self.game.reward()
            
        next_state = self._get_state(self.game)
        self.buffer.push(self.state, action, reward, next_state, abs(status))
        
        self.current_epoch_reward += reward
        self.epoch_steps += 1
        
        if status != 0 or self.epoch_steps >= 50:
            self.episode_rewards.append(self.current_epoch_reward)
            self.game = Gridworld(size=4, mode='random')
            self.state = self._get_state(self.game)
            self.current_epoch_reward = 0
            self.epoch_steps = 0
        else:
            self.state = next_state

    def training_step(self, batch, batch_idx):
        self.play_step()
        self.step_count += 1
        
        if self.step_count % 100 == 0:
            self.target_net.load_state_dict(self.model.state_dict())
            
        if len(self.buffer) < self.hparams.batch_size:
            return None
            
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(self.hparams.batch_size)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        q_values = self(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Double DQN target calculation
            next_q_primary = self.model(next_state_batch)
            best_actions = torch.argmax(next_q_primary, dim=1)
            next_q_target = self.target_net(next_state_batch)
            next_q_value = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
        expected_q_value = reward_batch + self.hparams.gamma * next_q_value * (1 - done_batch)
        
        loss = self.criterion(q_value, expected_q_value.detach())
        self.train_losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Linear decay matching repo
        if self.epsilon > self.hparams.epsilon_min:
            self.epsilon -= (1.0 / self.trainer.max_epochs)
            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, sample_size=100)
        return DataLoader(dataset, batch_size=1)

if __name__ == "__main__":
    print("Training Complete Dueling Double DQN (with Wall Penalty) in Random mode...")
    model = DQNLightning()
    
    trainer = pl.Trainer(
        max_epochs=1000,
        gradient_clip_val=1.0, 
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False
    )
    
    trainer.fit(model)
    
    plt.figure(figsize=(10, 7))
    plt.plot(model.train_losses, color='green')
    plt.title('Dueling DQN Training Loss (Random Mode)', fontsize=13)
    plt.xlabel('Steps', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    
    plt.savefig('hw3_3_results.png')
    print("Saved plot to hw3_3_results.png")
