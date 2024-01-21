import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import time

from agent import Agent
import os

# Frame rate settings
fps = 30
frame_duration = 1.0 / fps  # Duration of each frame in seconds

# Static model path
model_path = "C:\Super-Mario-Bros-RL\models"

# Model file name
ckpt_name = "model_main.pt"

# Construct the full file path
model_file_path = os.path.join(model_path, ckpt_name)

# Environment Setup
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = False  # Disable training
DISPLAY = True

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load the model
if os.path.isfile(model_file_path):
    print(f"Loading saved model from {model_file_path}")
    agent.load_model(model_file_path)
else:
    print("No saved model found at:", model_file_path)

# Main Loop for Demo
try:
    while True:  # Run indefinitely for demo
        state, _ = env.reset()
        done = False
        while not done:
            start_time = time.time()  # Record start time of the frame

            action = agent.choose_action(state)
            state, reward, done, truncated, info = env.step(action)

            # Calculate elapsed time and sleep if necessary
            elapsed = time.time() - start_time
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)

except KeyboardInterrupt:
    print("Exiting the demo...")
    env.close()
