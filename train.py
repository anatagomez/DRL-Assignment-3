import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the Agent class from student_agent.py
from student_agent import Agent, DQN

# Set seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
LEARNING_RATE = 0.0001
MEM_CAPACITY = 100000
UPDATE_FREQUENCY = 8
REWARD_CLIP = True
FRAME_SKIP = 4   


# Experience Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # determines how much prioritization is used
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame_idx = 0
        
    def beta_by_frame(self, frame_idx):
        return min(self.beta_end, self.beta_start + frame_idx * (self.beta_end - self.beta_start) / self.beta_frames)
    
    def push(self, *args):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        
        # Get sampling probabilities from priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame_idx)
        self.frame_idx += 1
        
        # Compute importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory)

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the correct output size from convolutional layers
        conv_out_size = self._get_conv_out(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _get_conv_out(self, shape):
        # Pass a dummy tensor through conv layers to get output shape
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # Make sure input has the right shape
        if len(x.size()) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        conv_out = self.conv(x).view(x.size()[0], -1)
        
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine value and advantage to get Q values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

def select_action(state, policy_net, n_actions, steps_done):
    """
    Epsilon-greedy action selection
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            return q_values.max(1)[1].item()
    else:
        return random.randrange(n_actions)

def optimize_model(policy_net, target_net, optimizer, memory, prioritized=False):
    """
    Performs one step of optimization
    """
    if len(memory) < BATCH_SIZE:
        return 0  # Not enough samples
    
    if prioritized:
        transitions, indices, weights = memory.sample(BATCH_SIZE)
    else:
        transitions = memory.sample(BATCH_SIZE)
        weights = torch.ones(BATCH_SIZE).to(device)
    
    # Convert batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))
    
    # Create tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                 device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    if non_final_next_states.size(0) == 0:
        return 0  # Skip optimization if no non-final states

    
    # Compute Q(s_t, a)
    q_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    # Double DQN: Select action using policy network, but evaluate using target network
    with torch.no_grad():
        next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()
    
    # Compute the expected Q values: Bellman equation
    reward_batch = reward_batch.squeeze()
    expected_q_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute TD error and loss
    td_error = (q_values - expected_q_values.unsqueeze(1)).abs()
    
    # Huber loss
    loss = nn.SmoothL1Loss(reduction='none')(q_values, expected_q_values.unsqueeze(1))
    weighted_loss = (loss * weights.unsqueeze(1)).mean()
    
    # Optimize the model
    optimizer.zero_grad()
    weighted_loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()
    
    # Update priorities in memory
    if prioritized:
        priorities = td_error.detach().cpu().numpy() + 1e-5  # small constant to avoid zero priority
        memory.update_priorities(indices, priorities)
    
    return weighted_loss.item()

def create_agent_with_training_networks(agent_object):
    """
    Create a version of the agent that uses the policy and target networks for training
    """
    # Create a training version of the agent
    agent = agent_object
    
    # Get input shape and action space from agent
    input_shape = agent.input_shape
    n_actions = agent.n_actions
    
    # Create training networks
    policy_net = DuelingDQN(input_shape, n_actions).to(device)
    target_net = DuelingDQN(input_shape, n_actions).to(device)
    
    # Initialize target network with same weights as policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is not trained directly
    
    return policy_net, target_net, input_shape, n_actions

def train():
    # Create Super Mario environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Create agent and get networks for training
    agent_object = Agent()
    policy_net, target_net, input_shape, n_actions = create_agent_with_training_networks(agent_object)
    
    # Use Adam optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # Use prioritized experience replay
    memory = PrioritizedReplayMemory(MEM_CAPACITY)
    
    # Training stats
    steps_done = 0
    episodes = 1000
    episode_rewards = []
    episode_lengths = []
    loss_values = []
    
    # Create directories for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Frame stacking for environment
    stacked_frames = deque(maxlen=agent_object.frame_stack)
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        
        # Process initial state
        frame = agent_object.preprocess_frame(state)
        # Initialize frame stack
        for _ in range(agent_object.frame_stack):
            stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=0).astype(np.float32) / 255.0
        stacked_state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)
        
        total_reward = 0
        done = False
        episode_step = 0
        episode_loss = 0
        
        # Record start time
        start_time = time.time()
        
        while not done:
            # Select action
            action = select_action(stacked_state_tensor, policy_net, n_actions, steps_done)

            total_skip_reward = 0
            for skip in range(FRAME_SKIP):
                if done:
                    break

                next_state, reward, done, info = env.step(action)
                steps_done += 1
                episode_step += 1

                # Clip reward if enabled
                if REWARD_CLIP:
                    reward = max(min(reward, 1), -1)

                total_skip_reward += reward

                # if done or episode_step >= MAX_STEPS_PER_EPISODE:
                #     break

                # Process next frame for stacking during skips
                if not done:
                    next_frame = agent_object.preprocess_frame(next_state)
                    stacked_frames.append(next_frame)

            total_reward += total_skip_reward

            # Prepare next state tensor
            if not done:
                next_stacked_state = np.stack(stacked_frames, axis=0).astype(np.float32) / 255.0
                next_stacked_state_tensor = torch.FloatTensor(next_stacked_state).unsqueeze(0).to(device)
            else:
                next_stacked_state_tensor = None

            # Store transition AFTER skipping
            memory.push(
                stacked_state_tensor,
                torch.tensor([[action]], device=device),
                next_stacked_state_tensor,
                torch.tensor([total_skip_reward], device=device),
                torch.tensor([done], device=device, dtype=torch.float32)
            )

            # Move to next state
            stacked_state_tensor = next_stacked_state_tensor

            # Optimize model
            if steps_done % UPDATE_FREQUENCY == 0:
                loss = optimize_model(policy_net, target_net, optimizer, memory, prioritized=True)
                if loss:
                    episode_loss += loss

            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log episode results
        avg_loss = episode_loss / episode_step if episode_step > 0 else 0
        loss_values.append(avg_loss)
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_step)
        
        if episode % 100 == 0:
            print(f"Episode {episode+1}/{episodes} | Steps: {episode_step} | Reward: {total_reward:.2f} | " 
                f"Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s | Total Steps: {steps_done}")
        
        # Save model checkpoints
        if (episode + 1) % 50 == 0:
            torch.save(policy_net.state_dict(), f'checkpoints/mario_model_episode_{episode+1}.pth')
            
            # Plot and save learning curve
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(1, 3, 2)
            plt.plot(episode_lengths)
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            plt.subplot(1, 3, 3)
            plt.plot(loss_values)
            plt.title('Average Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig(f'checkpoints/learning_curve_episode_{episode+1}.png')
            plt.close()
    
    # Save final model
    torch.save(policy_net.state_dict(), 'mario_model.pth')
    env.close()
    
    return policy_net

if __name__ == "__main__":
    train()