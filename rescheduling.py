import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from scheduler import SurgeryQuotaScheduler
import itertools
import logging


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


class Client:
    def __init__(self, name, urgency, completeness, complexity) -> None:
        self.name = name
        self.urgency = urgency
        self.completeness = completeness
        self.complexity = complexity
        self.acceptance_rate = np.random.randint(50, 76) / 100
        self._day = np.random.randint(0, 7)
        self._satisfied = False
        self._assigned_agent = None

    @property
    def appointment_day(self):
        return self._day

    @appointment_day.setter
    def appointment_day(self, value):
        self._day = value

    @property
    def satisfied(self):
        return self._satisfied

    @satisfied.setter
    def satisfied(self, value):
        self._satisfied = value

    @property
    def assigned_agent(self):
        return self._assigned_agent

    @assigned_agent.setter
    def assigned_agent(self, value):
        self._assigned_agent = value

    def give_feedback(self):
        answer = np.random.choice([True, False], p=[self.acceptance_rate, 1 - self.acceptance_rate])
        if answer == True:
            self.acceptance_rate = 1.0
        return answer


class MultiAgentSystemOperator:
    def __init__(self, list_of_clients) -> None:
        self.clients = list_of_clients

    def collect_feedback(self):
        for client in self.clients:
            client.satisfied = client.give_feedback()

    def assign_agnets(self, agents):
        for client in self.clients:
            health_state = (client.urgency, client.completeness, client.complexity)
            client.assigned_agent = agents[health_state]

    def get_actions(self, observaions):
        actions = {}
        for agent, client in zip(observations, self.clients):
            model = client.assigned_agent['model_file']
            actions[agent] = model.get_action(observaions[agent])
        return actions


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("logfile.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    urgency = range(1, 4)
    completeness = range(0, 2)
    complexity = range(0, 2)

    options = list(itertools.product(urgency, completeness, complexity))

    selected_states = np.random.choice(len(options), size=12, replace=False)
    health_states = [options[i] for i in selected_states]

    patiens = [Client(name=f'client_{i}',
                      urgency=np.random.randint(1, 4),
                      completeness=np.random.randint(0, 2),
                      complexity=np.random.randint(0, 2)) for i in range(100)]

    agents = {f'agent_{i}': MAPPOAgent(obs_dim=11, action_dim=3) for i in range(len(health_states))}

    for i, agent in enumerate(agents):
        agents[agent].load_model(f'trained_model/actor_{i}.pth')

    manager = MultiAgentSystemOperator(list_of_clients=patiens)
    manager.assign_agnets(
        {health_state: {'agent_name': agent_name, 'model_file': model_file} for health_state, (agent_name, model_file)
         in zip(health_states, agents.items())})

    e = 0

    while not all([client.satisfied for client in manager.clients]):

        logger.info(f"Starting episode {e}")

        env = SurgeryQuotaScheduler(render_mode='terminal', max_agents=len(patiens),
                                    max_days=7, max_episode_length=7)
        observations, _ = env.reset(
            options={
                'target_state': {0: {'min': 3, 'max': 3},
                                 1: {'min': 2, 'max': 2},
                                 2: {'min': 2, 'max': 2},
                                 3: {'min': 2, 'max': 2},
                                 4: {'min': 1, 'max': 1},
                                 5: {'min': 1, 'max': 1},
                                 6: {'min': 1, 'max': 1}},
                'agents_data': {f'agent_{i}': {'active': ~client.satisfied,
                                               'position': client.appointment_day if client.satisfied else np.random.randint(
                                                   0, 7),
                                               'base_reward': 1.0,
                                               'window': 5,
                                               'alpha': 2.0,
                                               'urgency': client.urgency,
                                               'completeness': client.completeness,
                                               'complexity': client.complexity,
                                               'mutation_rate': 0.0}
                                for i, client in enumerate(manager.clients)
                                }
            }
        )

        logger.info(f"Episode {e} - {env.render()}")

        while True:
            actions = manager.get_actions(observaions=observations)
            logger.debug(f"Actions: {actions}")
            observations, _, dones, truncations, _ = env.step(actions)

            for agent, client in zip(observations, manager.clients):
                client.appointment_day = observations[agent][3]

            logger.info(f"Episode {e} - {env.render()}")

            if any(dones.values()) or any(truncations.values()):
                break

        env.close()

        manager.collect_feedback()

        e += 1
