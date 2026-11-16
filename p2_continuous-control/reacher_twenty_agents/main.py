import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import numpy as np
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from agent import Agent, PrioritizedReplayBuffer
from utils import init_logger, log_action_distribution_continuous, log_gradient_norms, log_step_metrics, log_episode_metrics, close_logger, log_per_distribution_plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train an agent on the Reacher environment.")
parser.add_argument("--unity_worker_id", type=int, default=0, help="Unity environment worker ID.")
parser.add_argument("--unity_seed", type=int, default=0, help="Unity environment seed.")
parser.add_argument("--algorithm", type=str, default="TD3", choices=["DDPG", "TD3", "SAC"], help="The RL algorithm to use.")
parser.add_argument("--n_episodes", type=int, default=1000, help="Maximum number of training episodes.")
parser.add_argument("--max_t", type=int, default=1000, help="Maximum number of timesteps per episode.")
parser.add_argument("--prioritized_replay", action="store_true", help="Use prioritized experience replay.")
parser.add_argument("--buffer_size", type=int, default=int(1e5), help="Replay buffer capacity.")
parser.add_argument("--batch_size", type=int, default=128, help="Training minibatch size.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--tau", type=float, default=5e-3, help="Soft-update interpolation factor.")
parser.add_argument("--lr_actor", type=float, default=1e-4, help="Actor learning rate.")
parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate.")
parser.add_argument("--lr_step_size", type=int, default=50, help="Episodes between learning-rate decay steps.")
parser.add_argument("--lr_gamma", type=float, default=0.95, help="Learning-rate decay multiplier.")
parser.add_argument("--update_every", type=int, default=20, help="Environment steps between learning updates.")
parser.add_argument("--num_updates", type=int, default=10, help="Learning updates per update interval.")
parser.add_argument("--policy_delay", type=int, default=2, help="TD3 actor update delay.")
parser.add_argument("--target_noise", type=float, default=0.2, help="TD3 target policy smoothing noise.")
parser.add_argument("--noise_clip", type=float, default=0.5, help="TD3 target noise clipping limit.")
parser.add_argument("--per_alpha", type=float, default=0.6, help="Prioritized replay exponent.")
parser.add_argument("--beta_start", type=float, default=0.4, help="Initial importance-sampling correction.")
parser.add_argument("--beta_frames", type=int, default=5000, help="Steps over which replay beta reaches 1.")
parser.add_argument("--device", type=str, default="cpu", help="Torch device used for training.")


def main():
    args = parser.parse_args()
    env = UnityEnvironment(file_name="p2_continuous-control/reacher_twenty_agents/Reacher_Windows_x86_64/Reacher.exe", worker_id=args.unity_worker_id, seed=args.unity_seed)

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
        algorithm=args.algorithm.upper(), 
        use_prioritized_replay=args.prioritized_replay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        update_every=args.update_every,
        num_updates=args.num_updates,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
        per_alpha=args.per_alpha,
        beta_start=args.beta_start,
        beta_frames=args.beta_frames,
        device=args.device,
        )

    n_episodes = args.n_episodes                  # n_episodes (int): maximum number of training episodes
    max_t = args.max_t                      # max_t (int): maximum number of timesteps per episode
    scores = []                       # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores

    logger = init_logger("p2_continuous-control/reacher_twenty_agents/runs", "reacher_target")
    step_count = 0  # Globaler Schrittzähler für die X-Achse im TensorBoard

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                # Shape: (20, 33)
        agent_scores = np.zeros(num_agents)                  # Track score for each of the 20 agents

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
            torch.save(agent.local_net.state_dict(), Path(__file__).parent / f'checkpoint_{args.algorithm.lower()}.pth')
            break

        close_logger(logger)

    env.close()
        
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'p2_continuous-control/reacher_twenty_agents/reports/reacher_twenty_{args.algorithm.lower()}_score.png')
    #plt.show()


if __name__ == "__main__":
    main()