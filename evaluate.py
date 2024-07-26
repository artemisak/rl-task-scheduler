import argparse
from datetime import datetime

import numpy as np
import torch

from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from scheduler import SurgeryQuotaScheduler
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_capacity", type=int, default=3, help='max capacity of the environment')
    parser.add_argument("--max_agents", type=int, default=21, help='max number of agents')
    parser.add_argument("--max_days", type=int, default=7, help='planning horizon')
    parser.add_argument("--max_episode_length", type=int, default=7, help='max number of game turns')
    parser.add_argument("--algorithm", type=str, default="MADDPG", help="agilerl algorithm")
    parser.add_argument("--timestamp", type=str, default=f"{datetime.now()}", help='timestamp for process tracking')
    args = parser.parse_args()
    return args

def calculate_deviation(dict_episodes):
    ideal_distribution = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}

    average_distribution = {i: 0 for i in range(7)}
    for episode in dict_episodes:
        for key, value in episode.items():
            average_distribution[key] += value

    num_episodes = len(dict_episodes)
    for key in average_distribution:
        average_distribution[key] /= num_episodes

    deviation = {}
    for key in average_distribution:
        deviation[key] = abs(average_distribution[key] - ideal_distribution[key])

    percentage_deviation = {key: (value / 3) * 100 for key, value in deviation.items()}

    average_deviation = sum(percentage_deviation.values()) / len(percentage_deviation)

    print("Average Distribution:", average_distribution)
    print("Percentage Deviation:", percentage_deviation)
    print("Average Deviation across all days:", average_deviation)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = SurgeryQuotaScheduler(render_mode="terminal", max_capacity=args.max_capacity,
                                max_agents=args.max_agents, max_days=args.max_days,
                                max_episode_length=args.max_episode_length)
    channels_last = False  # Needed for environments that use images as observations
    env.reset()
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]

    # Pre-process image dimensions for pytorch convolutional layers
    if channels_last:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents
    last_episodes = []

    # Load the saved agent
    # path = f"./models/{args.algorithm}_{args.timestamp}/trained_agent.pt"
    path = f"./models/MADDPG_20240720_080859/trained_agent.pt"

    if args.algorithm == "MADDPG":
        model = MADDPG.load(path, device)
    else:
        model = MATD3.load(path, device)

    # Define test loop parameters
    episodes = 1000  # Number of episodes to test agent on
    max_steps = args.max_episode_length  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect individual agent rewards

    # Test loop for inference
    print("Evaluation...")
    for ep in tqdm(range(episodes)):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            if channels_last:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                    for agent_id, s in state.items()
                }

            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = model.get_action(
                state,
                training=False,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
            )
            if model.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Take action in environment
            state, reward, termination, truncation, info = env.step(action)

            if max(termination.values()) or max(truncation.values()):
                last_episodes.append(env.render())

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward for each agent
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

    calculate_deviation(last_episodes)
    env.close()
