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
    env = UnityEnvironment(file_name="p2_continuous-control/reacher_twenty_agents/Reacher_Windows_x86_64/Reacher.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    # number of agents in the environment
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    print('States look like:', states)
    state_size = states.shape[1]
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

    n_episodes = 500                  # n_episodes (int): maximum number of training episodes
    max_t = 1000                      # max_t (int): maximum number of timesteps per episode
    scores = []                       # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores

    logger = init_logger("p2_continuous-control/reacher_twenty_agents/runs", "reacher_target")
    step_count = 0  # Globaler Schrittzähler für die X-Achse im TensorBoard

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                # Shape: (20, 33)
        agent_scores = np.zeros(num_agents)                        # Track score for each of the 20 agents

        steps_on_target = 0
        episode_actions = []
        episode_steps = 0

        for t in range(max_t):
            actions = np.asarray(agent.act(states, add_noise=True), dtype=np.float32).reshape(num_agents, action_size)
            episode_actions.append(actions) # Track action choice

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            dones = np.array(env_info.local_done)

            steps_on_target += np.sum(rewards > 0)

            # in case of timeout, consider last Q state valid: rewards + gamma * Q_targets_next
            time_out = (t == max_t - 1)

            loss, avg_q = agent.step(states, actions, rewards, next_states, dones)
            log_step_metrics(logger, loss, avg_q, step_count)

            if loss is not None:
                log_gradient_norms(logger, agent.local_net, step_count)

            states = next_states
            agent_scores += rewards
            step_count += 1
            episode_steps += 1

            if np.any(dones) or time_out:
                break 

        # Average score across all 20 agents for this episode
        mean_episode_score = float(np.mean(agent_scores))
        min_episode_score = float(np.min(agent_scores))
        max_episode_score = float(np.max(agent_scores))

        scores_window.append(mean_episode_score)       # save most recent score
        scores.append(mean_episode_score)              # save most recent score

        # Decay exploration noise & step learning rate schedulers once per episode
        noise_std = agent.local_net.exploration_noise.std if hasattr(agent.local_net, "exploration_noise") else 0.0
        actor_lr, critic_lr = agent.lr_step()  # Step the learning rate scheduler and get the current LR

        log_episode_metrics(logger, mean_episode_score, noise_std, i_episode)
        logger.add_scalar('Score/Min_Agent_Score', min_episode_score, i_episode)
        logger.add_scalar('Score/Max_Agent_Score', max_episode_score, i_episode)
        logger.add_scalar('Behavioral/Avg_Steps_On_Target', steps_on_target / num_agents, i_episode)
        log_action_distribution_continuous(logger, episode_actions, i_episode)
        logger.add_scalar('Hyperparameters/Actor_LR', actor_lr, i_episode)
        logger.add_scalar('Hyperparameters/Critic_LR', critic_lr, i_episode)

        # Log PER priority distribution every 50 episodes
        if i_episode % 50 == 0 and isinstance(agent.memory, PrioritizedReplayBuffer):
            current_priorities = agent.memory.get_active_priorities()
            log_per_distribution_plot(logger, current_priorities, agent.memory.alpha, step_count)

        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tActor LR: {:.6f}\tCritic LR: {:.6f}'.format(
                i_episode, np.mean(scores_window), mean_episode_score, actor_lr, critic_lr
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print(
                '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tActor LR: {:.6f}\tCritic LR: {:.6f}'.format(
                    i_episode, np.mean(scores_window), mean_episode_score, actor_lr, critic_lr
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
    plt.savefig('p2_continuous-control/reacher_twenty_agents/reports/reacher_twenty_local_avg_score.png')
    #plt.show()


if __name__ == "__main__":
    main()