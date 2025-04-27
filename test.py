import torch
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from student_agent import Agent
import time

def test_agent(episodes=5, render=True):
    """
    Test the trained agent for a number of episodes
    """
    # Create Super Mario environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Create agent
    agent = Agent()
    
    # Test for multiple episodes
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        start_time = time.time()
        
        # Reset agent's frame buffer
        agent.initialized = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.02)  # Slow down rendering
            
            # Get action from agent
            action = agent.act(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Move to next state
            state = next_state
        
        elapsed_time = time.time() - start_time
        
        print(f"Episode {episode+1}: Reward = {total_reward}, Steps = {steps}, Time = {elapsed_time:.2f}s")
        total_rewards.append(total_reward)
    
    env.close()
    
    # Print average reward
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward

if __name__ == "__main__":
    test_agent(episodes=5, render=True)