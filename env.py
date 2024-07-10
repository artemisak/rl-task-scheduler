import sys
import time
import random
import functools
import pygame
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
from names_dataset import NameDataset
from scipy.stats import kstest, levy_stable

C = 4
N = 27
N_DAYS = 14
NUM_ITERS = int((N ** 2) / (N_DAYS * C))
MOVES = {0: 1, 1: -1, 2: 0}

nd = NameDataset()
popular_first_names = nd.get_top_names(country_alpha2='US', n=100)
popular_surnames = nd.get_top_names(country_alpha2='US', n=100,
                                    use_first_names=False)

def get_random_name():
    if random.randint(0, 1) == 0:
        return (random.choice(popular_first_names['US']['M']) + ' '
                + random.choice(popular_surnames['US']))
    else:
        return (random.choice(popular_first_names['US']['F']) + ' '
                + random.choice(popular_surnames['US']))

def generate_normal_values(N, mean=0.5, std_dev=0.1, low=0, high=1):
    values = np.random.normal(mean, std_dev, N)
    values = np.clip(values, low, high)
    return values

def create_rewards(threshold=0.999, mean=0.5, std_dev=0.1):
    p_value = 0
    attempts = 0
    while p_value < threshold:
        rewards = generate_normal_values(N, mean=0.5, std_dev=0.1, low=0, high=1)
        _, p_value = kstest(rewards, 'norm', args=(mean, std_dev))
        attempts += 1
    return list(rewards)

basic_rewards = create_rewards(threshold=0.999, mean=0.5, std_dev=0.1)

def generate_levy_distribution_number():
    alpha = 1.5
    beta = 0
    levy_numbers = levy_stable.rvs(alpha, beta, size=10000)
    threshold = np.percentile(levy_numbers, 95)
    tail_numbers = levy_numbers[levy_numbers > threshold]
    return np.random.choice(tail_numbers)

basic_rewards.append(generate_levy_distribution_number())

agents_rewards = dict()
for i in range(N + 1):
    agents_rewards.setdefault(get_random_name(), basic_rewards[i])

def draw_boxes(win, win_width, win_height, calendar):
    win.fill((0, 0, 0))
    box_width = win_width // 16
    box_height = 45
    center_x = win_width // 2
    center_y = win_height // 2
    for day, n in calendar.items():
        if n < 4:
            color = (0, 255, 0)
        elif n == 4:
            color = (255, 165, 0)
        else:
            color = (255, 0, 0)
        x = center_x - box_width * 7 + box_width * day
        y = center_y - box_height * 1.5
        pygame.draw.rect(win, color, (x, y, box_width, box_height))
        font = pygame.font.Font(None, 21)
        text = font.render("Day " + str(day), 1, (255, 255, 255))
        text_x = x + box_width // 2 - text.get_width() // 2
        text_y = y - text.get_height() * 1.5
        win.blit(text, (text_x, text_y))
        text = font.render(str(n), 1, (10, 10, 10))
        text_x = x + box_width // 2 - text.get_width() // 2
        text_y = y + box_height // 2 - text.get_height() // 2
        win.blit(text, (text_x, text_y))
    pygame.display.flip()

def game(win, win_width, win_height, observations):
    draw_boxes(win, win_width, win_height, convert_to_calendar(observations))
    time.sleep(0.5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

def convert_to_calendar(observations):
    dates = {day: 0 for day in range(N_DAYS)}
    for date in observations.values():
        if 0 <= date < N_DAYS:
            dates[date] += 1
    return dates

# def reward_map(slot_occupancy, position, end_of_episode, k, action, name):
#     reward = 0
#     if action == 0:
#         reward = -(agents_rewards[name] * k - (slot_occupancy[position] - 1) * agents_rewards[name])
#     elif action == 1:
#         reward = agents_rewards[name] * k - (slot_occupancy[position] - 2) * agents_rewards[name]
#     elif action == 2:
#         reward = (slot_occupancy[position] - 4) * agents_rewards[name]
#     reward = round(reward, 1)
#     return -reward if reward != -0.0 else 0.0

def reward_map(slot_occupancy, position, end_of_episode, k, action, name):
    reward = 0
    if action == 0:
        reward = agents_rewards[name] * k + (slot_occupancy[position] - 4) * agents_rewards[name]
    elif action == 1:
        reward = -agents_rewards[name] * k + (slot_occupancy[position] - 4) * agents_rewards[name]
    elif action == 2:
        reward = -agents_rewards[name] * (slot_occupancy[position] - 4)
    reward = round(reward, 1)
    return reward if reward != -0.0 else 0.0

class SurgeryQuotaScheduler(ParallelEnv):
    metadata = {'render_modes': ['human'],
                'name': 'sqsc_v1'}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.num_moves = 0
        self.possible_agents = list(agents_rewards.keys())
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.np_random = np.random.default_rng()
        self.agents = self.possible_agents.copy()
        self.agent_data = {agent: {} for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([3, 2, 2, 13, 29, 29, 29])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def reset(self, seed=None, options=None):
        self.render()
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.agents = self.possible_agents.copy()
        a = 0
        for agent in self.agents:
            self.agent_data[agent] = {
                'name': list(agents_rewards.keys())[a],
                'urgency': self.np_random.integers(1, 4),
                'completeness': self.np_random.integers(0, 2),
                'complexity': self.np_random.integers(0, 2),
                'position': self.np_random.integers(0, 14),
                'mutation_rate': 0
            }
            self.agent_data[agent]['k'] = (self.agent_data[agent]['completeness'] +
                                           (1 - self.agent_data[agent]['complexity'])) * self.agent_data[agent]['urgency']
            a += 1

        self.num_moves = 0
        slot_occupancy = self.convert_to_calendar({agent: self.agent_data[agent]['position'] for agent in self.agents})

        observations = {agent: np.array([
            self.agent_data[agent]['urgency'],
            self.agent_data[agent]['completeness'],
            self.agent_data[agent]['complexity'],
            self.agent_data[agent]['position'],
            slot_occupancy.get(self.agent_data[agent]['position'] - 1, 0),
            slot_occupancy.get(self.agent_data[agent]['position'], 0),
            slot_occupancy.get(self.agent_data[agent]['position'] + 1, 0)
        ], dtype=np.float32) for agent in self.agents}

        infos = {agent: {"num_moves": self.num_moves, "slot_occupancy": slot_occupancy} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.num_moves += 1

        if (self.num_moves >= NUM_ITERS - 1) and (
                sum(1 for action in actions.values() if action == 2) / len(self.agents) >= 0.8):
            terminations = {agent: True for agent in self.agents}
        else:
            terminations = {agent: False for agent in self.agents}

        if self.num_moves == NUM_ITERS * 2 - 1:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        for agent in self.agents:
            self.agent_data[agent]['position'] += MOVES[int(actions[agent])]
            if self.agent_data[agent]['position'] >= N_DAYS:
                self.agent_data[agent]['position'] = N_DAYS - 1
            if self.agent_data[agent]['position'] < 0:
                self.agent_data[agent]['position'] = 0
            if self.agent_data[agent]['position'] > N_DAYS / 2:
                if int(actions[agent]) == 0:
                    if self.agent_data[agent]['mutation_rate'] != 1.0:
                        self.agent_data[agent]['mutation_rate'] = min(self.agent_data[agent]['mutation_rate'] + 0.025, 1.0)
                elif int(actions[agent]) == 1:
                    if self.agent_data[agent]['mutation_rate'] != 0.0:
                        self.agent_data[agent]['mutation_rate'] = max(self.agent_data[agent]['mutation_rate'] - 0.025, 0.0)
            else:
                self.agent_data[agent]['mutation_rate'] = 0.0

            if np.random.choice([1, 0], p=[self.agent_data[agent]['mutation_rate'], 1 - self.agent_data[agent]['mutation_rate']]) == 1:
                if self.agent_data[agent]['urgency'] != 3:
                    self.agent_data[agent]['urgency'] += 1

            if np.random.choice([1, 0], p=[self.agent_data[agent]['mutation_rate'] / 2, 1 - self.agent_data[agent]['mutation_rate'] / 2]) == 1:
                if self.agent_data[agent]['complexity'] != 1:
                    self.agent_data[agent]['complexity'] = 1

            if np.random.choice([1, 0], p=[min(self.agent_data[agent]['mutation_rate'] * 2, 1),
                                           1 - min(self.agent_data[agent]['mutation_rate'] * 2, 1)]) == 1:
                if self.agent_data[agent]['completeness'] != 1:
                    self.agent_data[agent]['completeness'] = 1

            self.agent_data[agent]['k'] = max(1, (self.agent_data[agent]['complexity'] + (1 - self.agent_data[agent]['completeness'])) * self.agent_data[agent]['urgency'])

        slot_occupancy = convert_to_calendar({agent: self.agent_data[agent]['position'] for agent in self.agents})

        rewards = {agent: reward_map(slot_occupancy, self.agent_data[agent]['position'],
                                     max(max(terminations.values()), max(truncations.values())),
                                     self.agent_data[agent]['k'], actions[agent], self.agent_data[agent]['name']) for agent in self.agents}

        observations = {agent: np.array([
            self.agent_data[agent]['urgency'],
            self.agent_data[agent]['completeness'],
            self.agent_data[agent]['complexity'],
            self.agent_data[agent]['position'],
            slot_occupancy.get(self.agent_data[agent]['position'] - 1, 0),
            slot_occupancy.get(self.agent_data[agent]['position'], 0),
            slot_occupancy.get(self.agent_data[agent]['position'] + 1, 0)
        ], dtype=np.float32) for agent in self.agents}

        self.state = observations
        infos = {agent: {} for agent in self.possible_agents if agent in self.agents}

        self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "human":
            pygame.init()
            win_width, win_height = 1000, 600
            win = pygame.display.set_mode((win_width, win_height))
            pygame.display.set_caption("Surgery Quota Scheduler")
            game(win, win_width, win_height, {agent: self.agent_data[agent]['position'] for agent in self.agents})
        else:
            if len(self.agents) == N:
                string = 'Current state: \n'
                for agent in self.agents:
                    string += f"{agent}: {self.state[agent]}, "
                    print(string)

                    # Print the array of occupied slots per day
                    calendar = self.convert_to_calendar({agent: self.agent_data[agent]['position'] for agent in self.agents})
                    slots_per_day = [calendar[day] for day in range(14)]
                    print("Occupied slots per day:", slots_per_day)
            else:
                string = 'Game over'


    def close(self):
        pygame.quit()
        self.win = None
        self.possible_agents = None
        self.agent_name_mapping = None
        self.render_mode = None
        self.agents = None
        self.num_moves = None
        self.state = None
        self.agent_data = None

    @staticmethod
    def convert_to_calendar(observations):
        dates = {day: 0 for day in range(14)}
        for date in observations.values():
            if 0 <= date < 14:
                dates[date] += 1
        return dates
