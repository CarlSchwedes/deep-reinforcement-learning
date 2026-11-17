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

parser = argparse.ArgumentParser(description="Train an agent on the Soccer environment.")
parser.add_argument("--unity_worker_id", type=int, default=0, help="Unity environment worker ID.")
parser.add_argument("--unity_seed", type=int, default=0, help="Unity environment seed.")
parser.add_argument("--algorithm", type=str, default="TD3", choices=["DDPG", "TD3", "SAC", "DISCRETE_SAC", "DISCRETE_TD3"], help="The RL algorithm to use.")
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
    env = UnityEnvironment(file_name="p3_collab-compet/soccer/Soccer_Windows_x86_64/Soccer.exe", worker_id=args.unity_worker_id, seed=args.unity_seed)

    # print the brain names
    print(env.brain_names)

    # set the goalie brain
    g_brain_name = env.brain_names[0]
    g_brain = env.brains[g_brain_name]
    print(f"Goalie brain action space type: {g_brain.vector_action_space_type}")

    # set the striker brain
    s_brain_name = env.brain_names[1]
    s_brain = env.brains[s_brain_name]
    print(f"Striker brain action space type: {s_brain.vector_action_space_type}")

    # reset the environment
    env_info = env.reset(train_mode=True)

    # number of agents
    num_g_agents = len(env_info[g_brain_name].agents)
    print('Number of goalie agents:', num_g_agents)
    num_s_agents = len(env_info[s_brain_name].agents)
    print('Number of striker agents:', num_s_agents)

    # number of actions
    g_action_size = g_brain.vector_action_space_size
    print('Number of goalie actions:', g_action_size)
    s_action_size = s_brain.vector_action_space_size
    print('Number of striker actions:', s_action_size)

    # examine the state space 
    g_states = env_info[g_brain_name].vector_observations
    g_state_size = g_states.shape[1]
    print('There are {} goalie agents. Each receives a state with length: {}'.format(g_states.shape[0], g_state_size))
    s_states = env_info[s_brain_name].vector_observations
    s_state_size = s_states.shape[1]
    print('There are {} striker agents. Each receives a state with length: {}'.format(s_states.shape[0], s_state_size))

    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    g_agent = Agent(
        state_size=g_state_size, 
        action_size=g_action_size,
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

    s_agent = Agent(
        state_size=s_state_size, 
        action_size=s_action_size,
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

    n_episodes = args.n_episodes      # n_episodes (int): maximum number of training episodes
    max_t = args.max_t                # max_t (int): maximum number of timesteps per episode
    b_scores = []                        # list containing scores from each episode
    r_scores = []                        # list containing scores from each episode
    blue_scores_window = deque(maxlen=100)
    red_scores_window = deque(maxlen=100)
    decisive_games_window = deque(maxlen=100)

    logger = init_logger("p3_collab-compet/soccer/runs", "soccer")
    step_count = 0  # Globaler Schrittzähler für die X-Achse im TensorBoard

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)
        g_states = env_info[g_brain_name].vector_observations # get initial state (goalise)
        s_states = env_info[s_brain_name].vector_observations # get initial state (strikers)

        # Index 0 = Blue Team, Index 1 = Red Team
        blue_score = 0
        red_score = 0
        episode_actions = []
        episode_steps = 0

        for t in range(max_t):
            g_actions = g_agent.act(g_states, add_noise=True)
            s_actions = s_agent.act(s_states, add_noise=True)
            episode_actions.append((g_actions, s_actions)) # Track action choice

            # Step both brains simultaneously via a dictionary
            actions_dict = {
                g_brain_name: g_actions,
                s_brain_name: s_actions,
            }
            env_info = env.step(actions_dict)

            g_next_states = env_info[g_brain_name].vector_observations
            g_rewards = np.array(env_info[g_brain_name].rewards)
            g_dones = np.array(env_info[g_brain_name].local_done)

            s_next_states = env_info[s_brain_name].vector_observations
            s_rewards = np.array(env_info[s_brain_name].rewards)
            s_dones = np.array(env_info[s_brain_name].local_done)

            g_loss, g_avg_q = g_agent.step(g_states, g_actions, g_rewards, g_next_states, g_dones)
            s_loss, s_avg_q = s_agent.step(s_states, s_actions, s_rewards, s_next_states, s_dones)

            # Track team rewards (Goalie + Striker on each team)
            blue_score += g_rewards[0] + s_rewards[0]
            red_score += g_rewards[1] + s_rewards[1]

            # in case of timeout, consider last Q state valid: rewards + gamma * Q_targets_next
            time_out = (t == max_t - 1)

            log_step_metrics(logger, g_loss, g_avg_q, step_count)
            log_step_metrics(logger, s_loss, s_avg_q, step_count)

            if g_loss is not None:
                log_gradient_norms(logger, g_agent.local_net, step_count)
            if s_loss is not None:
                log_gradient_norms(logger, s_agent.local_net, step_count)

            g_states = g_next_states
            s_states = s_next_states
            step_count += 1
            episode_steps += 1

            if np.any(g_dones) or np.any(s_dones) or time_out:
                break 

        blue_scores_window.append(blue_score)
        b_scores.append(blue_score)
        red_scores_window.append(red_score)
        r_scores.append(red_score)
        is_decisive = 1.0 if abs(blue_score - red_score) > 0.5 else 0.0
        decisive_games_window.append(is_decisive)

        # Decay exploration noise & step learning rate schedulers once per episode
        #g_noise_std = g_agent.local_net.exploration_noise.std if hasattr(g_agent.local_net, "exploration_noise") else 0.0
        #s_noise_std = s_agent.local_net.exploration_noise.std if hasattr(s_agent.local_net, "exploration_noise") else 0.0
        g_actor_lr, g_critic_lr = g_agent.lr_step()  # Step the learning rate scheduler and get the current LR
        s_actor_lr, s_critic_lr = s_agent.lr_step()  # Step the learning rate scheduler and get the current LR

        # log_episode_metrics(logger, decisive_games_window, g_noise_std, i_episode)
        # log_episode_metrics(logger, decisive_games_window, s_noise_std, i_episode)
        log_action_distribution_continuous(logger, episode_actions, i_episode)
        logger.add_scalar('Hyperparameters/g_actor_LR', g_actor_lr, i_episode)
        logger.add_scalar('Hyperparameters/g_critic_LR', g_critic_lr, i_episode)
        logger.add_scalar('Hyperparameters/s_actor_LR', s_actor_lr, i_episode)
        logger.add_scalar('Hyperparameters/s_critic_LR', s_critic_lr, i_episode)

        # Log PER priority distribution every 50 episodes
        if i_episode % 50 == 0 and isinstance(g_agent.memory, PrioritizedReplayBuffer):
            current_priorities = g_agent.memory.get_active_priorities()
            log_per_distribution_plot(logger, current_priorities, g_agent.memory.alpha, step_count)
        if i_episode % 50 == 0 and isinstance(s_agent.memory, PrioritizedReplayBuffer):
            current_priorities = s_agent.memory.get_active_priorities()
            log_per_distribution_plot(logger, current_priorities, s_agent.memory.alpha, step_count)
 
        print(
            f"\rEpisode {i_episode:4d} | "
            f"Blue Avg: {np.mean(blue_scores_window):+.3f} | "
            f"Red Avg: {np.mean(red_scores_window):+.3f} | "
            f"Goal Rate (Last 100): {np.mean(decisive_games_window) * 100:.1f}% | "
            f"Actor LR g/s: {g_actor_lr:.6f}/{s_actor_lr:.6f} | "
            f"Critic LR g/s: {g_critic_lr:.6f}/{s_critic_lr:.6f}", 
            end=""
        )
        if i_episode % 100 == 0:
            print(
                f"Episode {i_episode:4d} | "
                f"Blue Avg: {np.mean(blue_scores_window):+.3f} | "
                f"Red Avg: {np.mean(red_scores_window):+.3f} | "
                f"Goal Rate (Last 100): {np.mean(decisive_games_window) * 100:.1f}% | "
                f"Actor LR g/s: {g_actor_lr:.6f}/{s_actor_lr:.6f} | "
                f"Critic LR g/s: {g_critic_lr:.6f}/{s_critic_lr:.6f}"
            )

            torch.save(g_agent.local_net.state_dict(), Path(__file__).parent / f'checkpoint_goalie_{args.algorithm.lower()}.pth')
            torch.save(s_agent.local_net.state_dict(), Path(__file__).parent / f'checkpoint_stricker_{args.algorithm.lower()}.pth')

        close_logger(logger)

    env.close()
        
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(b_scores)), b_scores, label='Blue Score')
    plt.plot(np.arange(len(r_scores)), r_scores, label='Red Score')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(f'p3_collab-compet/soccer/reports/soccer_{args.algorithm.lower()}_score.png')


if __name__ == "__main__":
    main()