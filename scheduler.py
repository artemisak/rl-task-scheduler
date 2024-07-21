import functools

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
from scipy.stats import bernoulli, levy_stable


class SurgeryQuotaScheduler(ParallelEnv):
    metadata = {'render_modes': ['human', 'terminal'],
                'name': 'sqsc_v1'}

    def __init__(self, render_mode=None, max_capacity=3, max_agents=21, max_days=7, max_episode_length=7):
        self.max_capacity = max_capacity
        self.max_agents = max_agents
        self.max_days = max_days
        self.max_episode_length = max_episode_length
        self.possible_agents = ["agent_" + str(r) for r in range(self.max_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_action_mapping = {0: 1, 1: -1, 2: 0}
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([3, 2, 2, self.max_days, self.max_agents,
                              self.max_agents, self.max_agents,
                              self.max_agents, self.max_agents,
                              self.max_agents, self.max_agents])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        if self.render_mode == "human":
            pygame.init()
            win_width, win_height = 1000, 600
            win = pygame.display.set_mode((win_width, win_height))
            pygame.display.set_caption("Surgery Quota Scheduler")
            self.game(win, win_width, win_height, {agent: self.agents_data[agent]['position'] for agent in self.agents})
        elif self.render_mode == 'terminal':
            calendar = self.convert_to_calendar({agent: self.agents_data[agent]['position'] for agent in self.agents})
            return calendar
        else:
            gymnasium.logger.warn('You are calling render mode without specifying any render mode.')
            return

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def reset(self, seed=None, options=None):
        self.np_random = np.random.default_rng(seed if seed is not None else None)
        self.agents = self.possible_agents[:]
        self.agents_data = {agent: {'base_reward': abs(levy_stable.rvs(alpha=1.9, beta=0.0)) if bernoulli.rvs(0.25) == 1 else abs(levy_stable.rvs(alpha=2.0, beta=0.0)),
                                    'urgency': self.np_random.integers(1, 4, 1).item(),
                                    'completeness': self.np_random.integers(0, 2, 1).item(),
                                    'complexity': self.np_random.integers(0, 2, 1).item(),
                                    'position': self.np_random.integers(0, self.max_days, 1).item(),
                                    'mutation_rate': 0} for agent in self.agents}
        self.max_base_reward = np.max([self.agents_data[agent]['base_reward'] for agent in self.agents])
        for agent in self.agents:
            self.agents_data[agent]['scaling_factor'] = max(1, (self.agents_data[agent]['complexity'] + (1 - self.agents_data[agent]['completeness'])) * self.agents_data[agent]['urgency'])
        self.num_moves = 0
        slot_occupancy = self.convert_to_calendar({agent: self.agents_data[agent]['position'] for agent in self.agents})
        observations = {agent: np.array([self.agents_data[agent]['urgency'],
                                         self.agents_data[agent]['completeness'],
                                         self.agents_data[agent]['complexity'],
                                         self.agents_data[agent]['position'],
                                         slot_occupancy.get(self.agents_data[agent]['position'] - 3, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] - 2, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] - 1, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'], self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] + 1, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] + 2, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] + 3, self.max_agents)]) for agent in
                        self.agents}
        infos = {agent: self.agents_data[agent] for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        self.num_moves += 1
        for agent in self.agents:
            new_position = self.agents_data[agent]['position'] + self.agent_action_mapping[int(actions[agent])]
            self.agents_data[agent]['position'] = max(0, min(new_position, self.max_days - 1))

            if self.agents_data[agent]['position'] >= self.max_days:
                self.agents_data[agent]['position'] = self.max_days - 1
            if self.agents_data[agent]['position'] < 0:
                self.agents_data[agent]['position'] = 0
            if self.agents_data[agent]['position'] > (self.max_days // 2):
                if int(actions[agent]) == 0:
                    if self.agents_data[agent]['mutation_rate'] != 1.0:
                        self.agents_data[agent]['mutation_rate'] = min(self.agents_data[agent]['mutation_rate'] + 0.05,
                                                                       1.0)
                elif int(actions[agent]) == 1:
                    if self.agents_data[agent]['mutation_rate'] != 0.0:
                        self.agents_data[agent]['mutation_rate'] = max(self.agents_data[agent]['mutation_rate'] - 0.05,
                                                                       0.0)
        slot_occupancy = self.convert_to_calendar({agent: self.agents_data[agent]['position'] for agent in self.agents})

        terminations = {agent: self.num_moves >= self.max_episode_length for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        rewards = {agent: self.reward_map(self.agents_data[agent]['position'],
                                          self.agents_data[agent]['base_reward'],
                                          self.agents_data[agent]['scaling_factor'],
                                          slot_occupancy,
                                          {day: self.max_capacity for day in range(self.max_days)}) for agent in self.agents}

        observations = {agent: np.array([self.agents_data[agent]['urgency'],
                                         self.agents_data[agent]['completeness'],
                                         self.agents_data[agent]['complexity'],
                                         self.agents_data[agent]['position'],
                                         slot_occupancy.get(self.agents_data[agent]['position'] - 3, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] - 2, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] - 1, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'], self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] + 1, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] + 2, self.max_agents),
                                         slot_occupancy.get(self.agents_data[agent]['position'] + 3, self.max_agents)]) for agent in self.agents}

        infos = {agent: self.agents_data[agent] for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def reward_map(self, position, base_reward, scaling_factor, observed_state, ideal_state):
        discrepancy = {day: abs(observed_state[day] - ideal_state[day]) for day in ideal_state}
        weighted_discrepancy_sum = np.sum([1 / (abs(position - day) + 1) * discrepancy[day] for day in discrepancy])
        return - weighted_discrepancy_sum * base_reward / scaling_factor

    def convert_to_calendar(self, positions):
        return {day: list(positions.values()).count(day) for day in range(self.max_days)}

    def game(self, win, positions):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            win.fill((255, 255, 255))

            font = pygame.font.Font(None, 36)
            y = 50
            for agent, pos in positions.items():
                text = font.render(f"{agent}: Day {pos + 1}", True, (0, 0, 0))
                win.blit(text, (50, y))
                y += 30

            pygame.display.flip()

        pygame.quit()
