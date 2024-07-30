import os

import numpy as np
import torch
from tqdm import tqdm
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
from scheduler import SurgeryQuotaScheduler

if __name__ == '__main__':

    seed = 42
    rng = np.random.default_rng(seed if seed is not None else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MADDPG.load(f'./models/MADDPG_20240720_080859/trained_agent.pt', device)

    num_envs = 12
    env = SurgeryQuotaScheduler(render_mode='terminal', max_capacity=3,
                                max_agents=21, max_days=7,
                                max_episode_length=7)
    env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    env.reset()

    options = {agent: {'active': rng.choice([True, False], size=1).item(),
                       'base_reward': 1.0,
                       'window': rng.choice([3, 5, 7], size=1).item(),
                       'alpha': rng.choice([2.0, 1.9, 1.8], size=1).item(),
                       'alpha_decay': 0.0,
                       'urgency': rng.integers(1, 4, 1).item(),
                       'completeness': rng.integers(0, 2, 1).item(),
                       'complexity': rng.integers(0, 2, 1).item(),
                       'position': rng.integers(0, 7, 1).item(),
                       'mutation_rate': 0.0} for agent in env.agents}

    field_names = ['state', 'action', 'reward', 'next_state', 'done']
    memory = MultiAgentReplayBuffer(10_000,
                                    field_names=field_names,
                                    agent_ids=env.agents,
                                    device=device)
    
    step = 0
    max_steps = 10_000
    learning_delay = 0
    evo_steps = 100
    
    with tqdm(total=max_steps, desc='Direct policy optimization (DPO) concept proof simulation progress') as pbar:

        while step < max_steps:

            state, info = env.reset(seed=seed, options=options)

            for idx_step in range(evo_steps // num_envs):
                cont_actions, discrete_action = model.get_action(states=state, training=True)
                next_state, reward, termination, truncation, info = env.step(discrete_action)

                step += num_envs
                pbar.update(num_envs)

                memory.save_to_memory(state,
                                      cont_actions,
                                      reward,
                                      next_state,
                                      termination,
                                      is_vectorised=True)
                
                if model.learn_step > num_envs:
                    learn_step = model.learn_step // num_envs
                    if idx_step % learn_step == 0 and len(memory) >= model.batch_size and memory.counter > learning_delay:
                        experiences = memory.sample(model.batch_size)
                        model.learn(experiences)
                elif len(memory) >= model.batch_size and memory.counter > learning_delay:
                    for _ in range(num_envs // model.learn_step):
                        experiences = memory.sample(model.batch_size)
                        model.learn(experiences)

                state = next_state

                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        reset_noise_indices.append(idx)
                model.reset_action_noise(reset_noise_indices)
    
    os.makedirs('./models/MADDPG_20240720_080859', exist_ok=True)
    model.save_checkpoint('./models/MADDPG_20240720_080859/trained_agent_dpo.pt')
    env.close()
