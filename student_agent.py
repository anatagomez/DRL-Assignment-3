import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import os
import cv2
from ddqn import DuelingDQN
# Set device
device = torch.device("cpu")  # For leaderboard submission
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Agent:
    def __init__(self):
        # Frame stacking
        self.frame_stack = 4
        
        # Input shape: Stacked grayscale frames (4, 84, 84)
        self.input_shape = (self.frame_stack, 84, 84)
        
        # Number of actions in COMPLEX_MOVEMENT
        self.n_actions = 12
        
        # Initialize DQN networks
        self.policy_net = DuelingDQN(self.input_shape, self.n_actions).to(device)
        
        # Load pre-trained model if available
        model_path = 'mario_model.pth'
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        
        self.policy_net.eval()  # Set to evaluation mode
        
        # Initialize frame buffer for stacking
        self.frames = deque(maxlen=self.frame_stack)
        
        # Flag to track if agent has been initialized
        self.initialized = False
    
    def preprocess_frame(self, frame):
        frame = frame.mean(axis=2).astype(np.uint8)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame
    
    def stack_frames(self, frame):
        """Stack frames for temporal information"""
        frame = self.preprocess_frame(frame)
        
        if not self.initialized:
            # For the first frame, we stack the same frame 4 times
            for _ in range(self.frame_stack):
                self.frames.append(frame)
            self.initialized = True
        else:
            self.frames.append(frame)
        
        # Stack frames into a single array and normalize
        stacked_frames = np.stack(self.frames, axis=0).astype(np.float32) / 255.0
        return stacked_frames
    
    def act(self, observation):
        """Choose an action based on the observation"""
        # Stack and preprocess frames
        state = self.stack_frames(observation)
        
        # Convert to tensor and get network prediction
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        # Choose the action with the highest Q-value
        action = q_values.max(1)[1].item()
        
        return action