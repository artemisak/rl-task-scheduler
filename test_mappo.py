import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scheduler import SurgeryQuotaScheduler
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        return self.network(obs)


class MAPPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.actor = Actor(obs_dim, action_dim)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))
        self.actor.eval()

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()


class MAPPOTester:

    def __init__(self, env, n_agents, obs_dim, action_dim, options=None):
        self.env = env
        self.options = options
        self.n_agents = n_agents
        self.agents = [MAPPOAgent(obs_dim, action_dim) for _ in range(n_agents)]

    def calculate_deviation(self, observed_states, target_state):
        num_episodes = len(observed_states)
        bootstrap_average_distribution = {i: sum(episode[i] for episode in observed_states) / num_episodes for i in
                                          range(7)}

        bootstrap_average_percentage_deviations = {i: 0 for i in range(7)}
        mean_target_state = {day: (target_state[day]['max'] + target_state[day]['min']) / 2 for day in target_state}
        for key, value in bootstrap_average_distribution.items():
            bootstrap_average_percentage_deviations[key] = np.abs(mean_target_state[key] - value) / mean_target_state[
                key]

        average_bootstrap_deviation = np.mean(list(bootstrap_average_percentage_deviations.values()))
        std_bootstrap_deviation = np.std(list(bootstrap_average_percentage_deviations.values()))
        return average_bootstrap_deviation, std_bootstrap_deviation

    def test(self, n_episodes, max_steps, target_state):
        all_observed_states = []
        all_final_positions = []
        all_scaling_factors = []

        rand_e = [np.random.randint(0, n_episodes+1) for _ in range(100)]

        for e in tqdm(range(n_episodes), desc="Bootstrap testing..."):
            obs, info = self.env.reset(options=self.options)

            if e in rand_e:
                logger.info(f'Episode {e}')
                logger.info(f'Initial environment state: {self.env.render()}')
                logger.info(f'Beliefs: {obs}')

            episode_observed_states = []

            for step in range(max_steps):
                actions = {}
                for i, agent in enumerate(self.agents):
                    action = agent.get_action(obs[f"agent_{i}"])
                    actions[f"agent_{i}"] = action

                if e in rand_e:
                    logger.info(f'Intentions: {actions}')

                next_obs, rewards, dones, truncations, info = self.env.step(actions)

                if e in rand_e:
                    logger.info(f'Desires: {rewards}')
                    logger.info(f'Environment state on step {step}:', self.env.render())
                    logger.info(f'Beliefs: {next_obs}')

                if any(dones.values()) or any(truncations.values()):
                    episode_observed_states.append(self.env.observed_state)
                    all_final_positions.extend([agent_info['position'] for agent_info in info.values()])
                    all_scaling_factors.extend(
                        [max(1, (agent_info['complexity'] + (1 - agent_info['completeness'])) * agent_info['urgency'])
                         for agent_info in info.values()])

                obs = next_obs

                if all(dones.values()) or all(truncations.values()):
                    break

            all_observed_states.extend(episode_observed_states)

        mean_deviation, std_deviation = self.calculate_deviation(all_observed_states, target_state)
        avg_bids_per_day = {day: sum(state[day] for state in all_observed_states) / len(all_observed_states) for day in
                            range(7)}

        scaling_factor_positions = {}
        for sf, pos in zip(all_scaling_factors, all_final_positions):
            if sf not in scaling_factor_positions:
                scaling_factor_positions[sf] = []
            scaling_factor_positions[sf].append(pos)
        avg_position_per_scaling_factor = {sf: np.mean(positions) for sf, positions in scaling_factor_positions.items()}

        return mean_deviation, std_deviation, avg_bids_per_day, avg_position_per_scaling_factor

    def bootstrap_test(self, n_episodes, max_steps, target_state):
        mean_deviation, std_deviation, avg_bids, avg_positions = self.test(n_episodes, max_steps, target_state)

        logger.info(f'Operator preferences: {target_state}')
        logger.info(f'Average number of bids per day: {avg_bids}')
        logger.info(f'Percentage deviation: mean = {mean_deviation:.2%}, std = {std_deviation: .2%}')
        logger.info(f'Average position per scaling factor: {avg_positions}')

    def load_model(self, path):
        for i, agent in enumerate(self.agents):
            agent.load_model(os.path.join(path, f'actor_{i}.pth'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("test_mappo.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    env = SurgeryQuotaScheduler(render_mode='terminal')
    n_agents = 12
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    tester = MAPPOTester(env, n_agents, obs_dim, action_dim)
    tester.load_model(path='trained_model')
    tester.bootstrap_test(n_episodes=10_000, max_steps=7,
                          target_state={0: {'min': 3, 'max': 3},
                                        1: {'min': 2, 'max': 2},
                                        2: {'min': 2, 'max': 2},
                                        3: {'min': 2, 'max': 2},
                                        4: {'min': 1, 'max': 1},
                                        5: {'min': 1, 'max': 1},
                                        6: {'min': 1, 'max': 1}})
