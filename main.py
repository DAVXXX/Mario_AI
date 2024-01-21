import torch
import signal
import os
import shutil

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers


from utils import *

# Static model path
model_path = "C:\\Users\\pc\\Documents\\MarioAI\\Super-Mario-Bros-RL\\models"

# Model file name
ckpt_name = "model_main.pt"

# Construct the full file path
model_file_path = os.path.join(model_path, ckpt_name)


if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

#
def clear_temp_directory(temp_folder_path):
    for item in os.listdir(temp_folder_path):
        item_path = os.path.join(temp_folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')

# Usage
temp_folder = 'C:\\Users\\pc\\AppData\\Local\\Temp'
clear_temp_directory(temp_folder)
    
# Define a function to handle Ctrl+C
def signal_handler(signum, frame):
    print("Received Ctrl+C. Saving the model...")
    save_model_state(agent, model_path, "model_main.pt")
    print("Model state saved. Exiting...")
    env.close()
    exit()

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

    
def save_model_state(agent, model_path, filename="model_main.pt"):
    state = {
        "model_state": agent.online_network.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        # Add any other agent attributes you want to save, e.g., epsilon
        "epsilon": agent.epsilon
    }
    torch.save(state, os.path.join(model_path, filename))
    
def load_model_state(agent, model_path, filename="model_main.pt"):
    file_path = os.path.join(model_path, filename)
    if os.path.isfile(file_path):
        print(f"Loading saved model state from {file_path}")
        state = torch.load(file_path)
        agent.online_network.load_state_dict(state["model_state"])
        agent.optimizer.load_state_dict(state["optimizer_state"])
        # Load any other saved attributes
        agent.epsilon = state["epsilon"]
    else:
        print("No saved model state found. Starting from scratch.")

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load the model unconditionally
if os.path.isfile(model_file_path):
    print(f"Loading saved model from {model_file_path}")
    agent.load_model(model_file_path)
else:
    print("No saved model found at:", model_file_path)

# Enter training mode based on SHOULD_TRAIN
print("SHOULD_TRAIN is set to:", SHOULD_TRAIN)
if SHOULD_TRAIN:
    print("Training mode is ON.")
    try:
        for i in range(NUM_OF_EPISODES):
            print("Episode:", i)
            done = False
            state, _ = env.reset()
            total_reward = 0
            while not done:
                a = agent.choose_action(state)
                new_state, reward, done, truncated, info = env.step(a)
                total_reward += reward

                if SHOULD_TRAIN:
                    agent.store_in_memory(state, a, reward, new_state, done)
                    agent.learn()

                state = new_state

            print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:",
                  len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

            if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
                agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

            print("Total reward:", total_reward)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Saving the model and exiting...")
        agent.save_model(os.path.join(model_path, "model_interrupted.pt"))
        print("Model saved. Exiting...")
        env.close()
else:
    print("Training mode is OFF.")
    # Insert your evaluation or testing logic here


env.reset()
next_state, reward, done, trunc, info = env.step(action=0)


