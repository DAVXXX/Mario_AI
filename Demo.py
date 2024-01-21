import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent
from gym_super_mario_bros.actions import RIGHT_ONLY
import torch

# Path to the trained model
model_path = "C:\\Super-Mario-Bros-RL\\models\\model_main.pt"

# Initialize the game environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Determine the number of actions
num_actions = env.action_space.n

channels = 4  # Update this to match the training setup
height = 84
width = 84
input_dims = (channels, height, width)


# Initialize the agent with the appropriate parameters
agent = Agent(input_dims, num_actions)

# Load the trained model into the agent
agent.load_model(model_path)

# Demo loop
num_demo_episodes = 10  # Define the number of demo episodes
for episode in range(num_demo_episodes):
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, info = obs  # Unpack the observation and info if available
    else:
        info = {}  # Create an empty info dictionary

    done = False
    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, done, info = env.step(action)
        env.render()  # Render the game screen if possible
        obs = next_obs  # Update obs with next_obs
    print(f"Episode: {episode}, Score: {info.get('score', 'N/A')}")





env.close()
