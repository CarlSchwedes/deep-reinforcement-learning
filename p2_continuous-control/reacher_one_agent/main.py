import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import numpy as np
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from ppo_agent import Agent, PrioritizedReplayBuffer
from utils import init_logger, log_action_distribution_continuous, log_gradient_norms, log_step_metrics, log_episode_metrics, close_logger, log_per_distribution_plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    env = UnityEnvironment(file_name="p2_continuous-control/reacher_one_agent/Reacher_Windows_x86_64/Reacher.exe")

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

    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent = Agent(
        state_size=state_size, 
        action_size=action_size,
        algorithm="SAC", 
        use_prioritized_replay=True,
        device="cpu"
        )

    n_episodes = 1000     # n_episodes (int): maximum number of training episodes
    max_t = 1000          # max_t (int): maximum number of timesteps per episode
 
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    logger = init_logger("p2_continuous-control/reacher_one_agent/runs", "reacher_target")
    step_count = 0  # Globaler Schrittzähler für die X-Achse im TensorBoard
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0.0

        steps_on_target = 0
        episode_actions = []
        episode_steps = 0

        for t in range(max_t):
            action = agent.act(state)
            episode_actions.append(action) # Track action choice

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            if reward > 0:
                steps_on_target += 1

            # in case of timeout, consider last Q state valid: rewards + gamma * Q_targets_next
            time_out = (t == max_t - 1)

            loss, avg_q = agent.step(state, action, reward, next_state, done)
            log_step_metrics(logger, loss, avg_q, step_count)

            if loss is not None:
                log_gradient_norms(logger, agent.local_net, step_count)

            state = next_state
            score += reward
            step_count += 1
            episode_steps += 1

            if done or time_out:
                break 

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        # Decay exploration noise & step learning rate schedulers once per episode
        noise_std = agent.local_net.exploration_noise.std if hasattr(agent.local_net, "exploration_noise") else 0.0
        actor_lr, critic_lr = agent.lr_step()  # Step the learning rate scheduler and get the current LR

        log_episode_metrics(logger, score, noise_std, i_episode)
        logger.add_scalar('Behavioral/Steps_On_Target', steps_on_target, i_episode)
        logger.add_scalar('Behavioral/Target_Hold_Ratio', steps_on_target / max(1, episode_steps), i_episode)
        log_action_distribution_continuous(logger, episode_actions, i_episode)
        logger.add_scalar('Hyperparameters/Actor_LR', actor_lr, i_episode)
        logger.add_scalar('Hyperparameters/Critic_LR', critic_lr, i_episode)

        # Log PER priority distribution every 50 episodes
        if i_episode % 50 == 0 and isinstance(agent.memory, PrioritizedReplayBuffer):
            current_priorities = agent.memory.get_active_priorities()
            log_per_distribution_plot(logger, current_priorities, agent.memory.alpha, step_count)

        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tActor LR: {:.6f}\tCritic LR: {:.6f}'.format(
                i_episode, np.mean(scores_window), score, actor_lr, critic_lr
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print(
                '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tActor LR: {:.6f}\tCritic LR: {:.6f}'.format(
                    i_episode, np.mean(scores_window), score, actor_lr, critic_lr
                )
            )

        # Reacher is solved at an average score of +30.0 over 100 consecutive episodes
        if np.mean(scores_window) >= 30.0 and len(scores_window) == 100:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode - 100, np.mean(scores_window)
                )
            )
            torch.save(agent.local_net.state_dict(), Path(__file__).parent / 'checkpoint.pth')
            break

        close_logger(logger)
        
    env.close()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('p2_continuous-control/reacher_one_agent/reports/reacher_one_local_avg_score.png')
    #plt.show()


if __name__ == "__main__":
    main()