import torch
import numpy as np
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from dqn_agent import Agent
from utils import init_logger, log_step_metrics, log_episode_metrics, close_logger


def main():
    env = UnityEnvironment(file_name="p1_value_based_methods/navigation_pixel/Banana_Windows_x86_64/Banana.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    n_episodes=2000     # n_episodes (int): maximum number of training episodes
    max_t=1000          # max_t (int): maximum number of timesteps per episode
    eps_start=1.0       # eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end=0.01        # eps_end (float): minimum value of epsilon
    eps_decay=0.995     # eps_decay (float): multiplicative factor (per episode) for decreasing epsilon 

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    logger = init_logger("p1_value_based_methods/navigation_pixel/runs", "banana_collector")
    step_count = 0  # Globaler Schrittzähler für die X-Achse im TensorBoard
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = env.step(int(action))[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # in case of timeout, consider last Q state valid: rewards + gamma * Q_targets_next
            time_out = (t == max_t - 1)

            loss, avg_q = agent.step(state, action, reward, next_state, done)
            log_step_metrics(logger, loss, avg_q, step_count)

            state = next_state
            score += reward
            step_count += 1

            if done or time_out:
                break 

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        log_episode_metrics(logger, score, eps, i_episode)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

        close_logger(logger)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('p1_value_based_methods/reports/navigation_vector_score.png')
    plt.show()


if __name__ == "__main__":
    main()