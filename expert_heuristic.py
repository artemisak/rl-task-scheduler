import itertools
import logging
import numpy as np


class Client:
    def __init__(self, name, urgency, completeness, complexity):
        self.name = name
        self.urgency = urgency
        self.completeness = completeness
        self.complexity = complexity

    @property
    def scaling_factor(self):
        return max(1, (self.complexity + (1 - self.completeness)) * self.urgency)


class Manager:
    def __init__(self, planning_horizon, preferences):
        self.planning_horizon = planning_horizon
        self.preferences = preferences if preferences is not None else {
            day: {'min': np.min(numbers), 'max': np.max(numbers)}
            for day, numbers in enumerate(np.random.randint(1, 3, (7, 2)))
        }

    def reset_schedule(self):
        self.schedule = {day: [] for day in range(self.planning_horizon)}

    def manage(self, upcoming_requests):
        for request in upcoming_requests:
            for day in range(self.planning_horizon):
                if len(self.schedule[day]) < self.preferences[day]['max']:
                    self.schedule[day].append(request)
                    break
        return self.schedule


class Estimator:
    def __init__(self, schedules, target_state=None):
        self.observed_states = [{day: len(schedule[day]) for day in schedule.keys()} for schedule in schedules]
        self.schedules = schedules
        if target_state is not None:
            self.target_state = target_state
        else:
            self.target_state = {
                day: {'min': np.min(numbers), 'max': np.max(numbers)}
                for day, numbers in enumerate(np.random.randint(1, 3, (7, 2)))
            }

    def describe(self):
        num_episodes = len(self.observed_states)

        deviations = []
        total_clients_per_day = {i: 0 for i in range(len(self.schedules[0]))}
        all_scaling_factors = []
        all_final_positions = []

        for schedule in self.schedules:
            for day in schedule:
                actual_clients = len(schedule[day])
                total_clients_per_day[day] += actual_clients
                target_min = self.target_state[day]['min']
                target_max = self.target_state[day]['max']

                deviation = max(0, target_min - actual_clients) + max(0, actual_clients - target_max)
                deviations.append(deviation)

                for client in schedule[day]:
                    all_scaling_factors.append(client.scaling_factor)
                    all_final_positions.append(day)

        mean_deviation = np.mean(deviations) / num_episodes
        std_deviation = np.std(deviations) / num_episodes
        avg_clients_per_day = {day: total / num_episodes for day, total in total_clients_per_day.items()}

        scaling_factor_positions = {}
        for sf, pos in zip(all_scaling_factors, all_final_positions):
            if sf not in scaling_factor_positions:
                scaling_factor_positions[sf] = []
            scaling_factor_positions[sf].append(pos)
        avg_position_per_scaling_factor = {sf: np.mean(positions) for sf, positions in scaling_factor_positions.items()}

        logger.info(f'Operator preferences: {self.target_state}')
        logger.info(f'Average number of bids per day: {avg_clients_per_day}')
        logger.info(f'Percentage deviation: mean = {mean_deviation:.2%}, std = {std_deviation:.2%}')
        logger.info(f'Average position per scaling factor: {avg_position_per_scaling_factor}')

def run_simulation(n_episodes=10_000):
    planning_horizon = 7
    health_state = list(itertools.product(range(1, 4),
                                          range(0, 2),
                                          range(0, 2)))

    target_state = {
        0: {'min': 3, 'max': 3},
        1: {'min': 2, 'max': 2},
        2: {'min': 2, 'max': 2},
        3: {'min': 2, 'max': 2},
        4: {'min': 1, 'max': 1},
        5: {'min': 1, 'max': 1},
        6: {'min': 1, 'max': 1},
    }

    manager = Manager(planning_horizon, preferences=target_state)

    schedules = []

    for _ in range(n_episodes):
        manager.reset_schedule()
        clients = [Client(name=f'client_{i}', urgency=urgency, completeness=completeness, complexity=complexity)
                   for i, (urgency, completeness, complexity) in enumerate(health_state)]
        schedules.append(manager.manage(np.random.choice(clients, size=12, replace=True)))

    estimator = Estimator(schedules, target_state)
    estimator.describe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("expert_heuristic.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    run_simulation()
