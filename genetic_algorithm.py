import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import levy_stable
import itertools
from collections import defaultdict


class GeneticAlgorithmScheduler:
    def __init__(self, max_agents=12, max_days=7, population_size=100, generations=1000):
        self.max_agents = max_agents
        self.max_days = max_days
        self.population_size = population_size
        self.generations = generations
        self.agents_data = self.generate_agents_data()
        self.target_state = {day: self.max_agents / self.max_days for day in range(self.max_days)}

        self.log_dir = 'logs/genetic_algorithm'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def generate_agents_data(self):
        agents_data = {}
        rng = np.random.default_rng()
        for (urgency, complexity, completeness), agent in zip(
                itertools.product(range(1, 4), range(0, 2), range(0, 2)),
                [f"agent_{r}" for r in range(self.max_agents)]
        ):
            agents_data[agent] = {
                'active': True,
                'base_reward': 1.0,
                'window': 3,
                'alpha': 2.0,
                'urgency': urgency,
                'complexity': complexity,
                'completeness': completeness,
                'mutation_rate': 0.0,
                'position': rng.integers(0, 7, 1).item(),
                'scaling_factor': ((complexity + (1 - completeness)) * urgency)
            }
        return agents_data

    def create_individual(self):
        return np.array([self.agents_data[f"agent_{i}"]["position"] for i in range(self.max_agents)])

    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, individual):
        observed_state = {day: 0 for day in range(self.max_days)}
        for agent, day in enumerate(individual):
            observed_state[day] += 1

        fitness_score = 0
        for agent, day in enumerate(individual):
            agent_data = self.agents_data[f"agent_{agent}"]
            discrepancy = {d: abs(observed_state[d] - self.target_state[d]) for d in range(self.max_days)}
            window = [int((day + np.ceil(d - agent_data['window'] / 2)) % self.max_days) for d in
                      range(agent_data['window'])]
            masked_discrepancy = {d: discrepancy[d] if d in window else 0 for d in discrepancy}
            rv = levy_stable(agent_data['alpha'], 0.0, loc=day, scale=1.0)
            weighted_discrepancy_sum = np.sum([rv.pdf(d) * masked_discrepancy[d] for d in masked_discrepancy])
            reward = -weighted_discrepancy_sum * agent_data['base_reward']
            fitness_score += reward

        return fitness_score

    def selection(self, population, k=2):
        selected = np.random.choice(len(population), k, replace=False)
        return max([population[i] for i in selected], key=self.fitness)

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutation(self, individual, mutation_rate=0.01):
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] = np.random.randint(0, self.max_days)
        return individual

    def run(self):
        population = self.create_population()
        best_fitness_history = []

        for generation in tqdm(range(self.generations), leave=False):
            new_population = []

            for _ in range(self.population_size):
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)

            population = new_population

            best_individual = max(population, key=self.fitness)
            best_fitness = self.fitness(best_individual)
            best_fitness_history.append(best_fitness)

            with self.summary_writer.as_default():
                tf.summary.scalar('Best Fitness', best_fitness, step=generation)

        return max(population, key=self.fitness), best_fitness_history

    def get_schedule_details(self, best_individual):
        schedule_details = []
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for agent, day in enumerate(best_individual):
            agent_data = self.agents_data[f"agent_{agent}"]
            schedule_details.append({
                'agent': f"agent_{agent}",
                'day': days[day],
                'urgency': agent_data['urgency'],
                'completeness': agent_data['completeness'],
                'complexity': agent_data['complexity'],
                'scaling_factor': agent_data['scaling_factor']
            })

        return schedule_details


def run_experiment(num_runs=5):
    all_results = []
    all_fitness_histories = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        scheduler = GeneticAlgorithmScheduler()
        best_schedule, fitness_history = scheduler.run()
        schedule_details = scheduler.get_schedule_details(best_schedule)
        all_results.append(schedule_details)
        all_fitness_histories.append(fitness_history)

        print(f"Best fitness for run {run + 1}: {fitness_history[-1]}")
        for item in schedule_details:
            print(f"{item['agent']} placed on {item['day']}, urgency: {item['urgency']}, "
                  f"completeness: {item['completeness']}, complexity: {item['complexity']}, "
                  f"scaling_factor: {item['scaling_factor']}")

    return all_results, all_fitness_histories


def summarize_results(all_results, all_fitness_histories):
    num_runs = len(all_results)

    final_fitness_scores = [history[-1] for history in all_fitness_histories]
    avg_final_fitness = np.mean(final_fitness_scores)
    std_final_fitness = np.std(final_fitness_scores)

    print(f"\nSummary across {num_runs} runs:")
    print(f"Average final fitness: {avg_final_fitness:.2f} ± {std_final_fitness:.2f}")

    day_counts = defaultdict(lambda: defaultdict(int))
    for run_results in all_results:
        for item in run_results:
            day_counts[item['agent']][item['day']] += 1

    print("\nMost common day assignments:")
    for agent in day_counts:
        most_common_day = max(day_counts[agent], key=day_counts[agent].get)
        frequency = day_counts[agent][most_common_day]
        print(f"{agent}: {most_common_day} ({frequency}/{num_runs} runs)")

    convergence_points = []
    for history in all_fitness_histories:
        for i, fitness in enumerate(history):
            if i > 0 and abs(fitness - history[-1]) < 0.01 * abs(history[-1]):
                convergence_points.append(i)
                break
    avg_convergence = np.mean(convergence_points)
    print(f"\nAverage convergence generation: {avg_convergence:.2f}")


if __name__ == "__main__":
    all_results, all_fitness_histories = run_experiment(num_runs=5)
    summarize_results(all_results, all_fitness_histories)
