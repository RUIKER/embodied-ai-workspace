import numpy as np
from tqdm import tqdm


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_return = 0.0
            transition_dict = {
                'states': [], 'actions': [], 'next_states': [],
                'rewards': [], 'dones': [],
            }
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            agent.update(transition_dict)
            return_list.append(episode_return)
            if (episode + 1) % 10 == 0:
                progress.set_postfix({
                    'episode': episode + 1,
                    'return': '%.3f' % np.mean(return_list[-10:]),
                })
            progress.update(1)
    return return_list
