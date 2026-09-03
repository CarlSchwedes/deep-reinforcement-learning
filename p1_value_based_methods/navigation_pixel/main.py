import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from unityagents import UnityEnvironment

import random
import numpy as np
from pathlib import Path
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from dqn_agent import Agent, PrioritizedReplayBuffer
from utils import (
    init_logger, 
    log_action_distribution, 
    log_behavioral_metrics, 
    log_gradient_norms, 
    log_step_metrics, 
    log_episode_metrics, 
    close_logger, 
    log_per_distribution_plot, 
    log_td_error_metrics
    )


class FrameStacker:
    """Manages stacking of 4 consecutive frames for CNN temporal dynamics."""
    
    def __init__(self, stack_size=4, img_shape=(84, 84)):
        self.stack_size = stack_size
        self.img_shape = img_shape
        # Queue with a maximum capacity length of 4 entries
        self.frames = deque(maxlen=stack_size)
        
    def reset(self, initial_frame):
        """Called at the start of a new episode. Fills the stack with the first frame."""
        # Convert and secure the very first frame as a normalized float32 type array
        frame = np.array(initial_frame, dtype=np.float32) / 255.0
        
        # Initialize the stack completely using the first frame (4x copy of the same image)
        for _ in range(self.stack_size):
            self.frames.append(frame)
            
        return self.get_stack()
        
    def step(self, new_frame):
        """Called at each environment step. Drops oldest frame, appends the new one."""
        frame = np.array(new_frame, dtype=np.float32) / 255.0
        self.frames.append(frame)
        return self.get_stack()
        
    def get_stack(self):
        """Returns the stacked frames with shape (4, 84, 84)."""
        # Stacks the 4 frames along axis 0 (acting as channels for the CNN)
        return np.stack(self.frames, axis=0)
    

def preprocess_frame(raw_frame):
    """Converts the raw Unity environment observation into a 2D grayscale array.
    Expects raw_observation shape: (1, 84, 84, 3) or similar list format from Unity.
    """
    # 1. Remove the batch dimension: (1, 84, 84, 3) -> (84, 84, 3)
    if len(raw_frame.shape) == 4:
        img = raw_frame[0]
    else:
        img = raw_frame

    # 2. Convert RGB to grayscale (Standard ITU-R weights for luminosity tracking)
    # This reduces the array dimensions down from (84, 84, 3) to a flat (84, 84)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    grayscale_img = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # 3. Ensure that the final image array has the exact targeted shape of (84, 84)
    return np.array(grayscale_img, dtype=np.float32)


def main():
    env = UnityEnvironment(file_name="p1_value_based_methods/navigation_pixel/VisualBanana_Windows_x86_64/Banana.exe")

    stacker = FrameStacker(stack_size=4, img_shape=(84, 84))

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
    state = env_info.visual_observations[0]
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
        use_prioritized_replay=True,
        use_dueling_network=True,
        use_replay_start_size=True,
        use_noisy_nets=True,
        n_steps=3,
        use_distributional_rl=True
    )

    n_episodes = 2000     # n_episodes (int): maximum number of training episodes
    max_t = 1000          # max_t (int): maximum number of timesteps per episode
 
    eps_start = 0.0 if agent.use_noisy_nets else 1.0          # eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end = 0.0 if agent.use_noisy_nets else 0.01           # eps_end (float): minimum value of epsilon
    eps_decay = 1.0 if not agent.use_noisy_nets else 0.995    # eps_decay (float): multiplicative factor (per episode) for decreasing epsilon 

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    logger = init_logger("p1_value_based_methods/navigation_pixel/runs", "banana_collector")
    step_count = 0  # Globaler Schrittzähler für die X-Achse im TensorBoard

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        raw_frame = env_info.visual_observations[0]
        clean_frame = preprocess_frame(raw_frame)
        state = stacker.reset(clean_frame)
        score = 0

        yellow_collected = 0
        blue_collected = 0
        episode_actions = []
        episode_steps = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            episode_actions.append(action) # Track action choice

            env_info = env.step([action])[brain_name]
            raw_next_frame = env_info.visual_observations[0]
            clean_next_frame = preprocess_frame(raw_next_frame)
            next_state = stacker.step(clean_next_frame)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            if reward > 0:
                yellow_collected += 1
            elif reward < 0:
                blue_collected += 1

            # in case of timeout, consider last Q state valid: rewards + gamma * Q_targets_next
            time_out = (t == max_t - 1)

            loss, avg_q, batch_td_errors = agent.step(state, action, reward, next_state, done)
            log_step_metrics(logger, loss, avg_q, step_count)

            if batch_td_errors is not None:
                log_td_error_metrics(logger, batch_td_errors, step_count)

            if loss is not None:
                log_gradient_norms(logger, agent.qnetwork_local, step_count)

            state = next_state
            score += reward
            step_count += 1
            episode_steps += 1

            if done or time_out:
                break 

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        current_lr = agent.lr_step()  # Step the learning rate scheduler and get the current LR

        log_episode_metrics(logger, score, eps, i_episode)
        log_behavioral_metrics(logger, yellow_collected, blue_collected, episode_steps, i_episode)
        log_action_distribution(logger, episode_actions, i_episode)
        logger.add_scalar('Hyperparameters/Learning_Rate', current_lr, i_episode)

        # --- NEW: Log the visual distribution plot every 50 episodes ---
        if i_episode % 50 == 0 and isinstance(agent.memory, PrioritizedReplayBuffer):
            # Extract the raw priorities array from your PER instance
            current_priorities = agent.memory.get_active_priorities()
            current_alpha = agent.memory.alpha
            
            # Pass it to our new visual summary utility
            log_per_distribution_plot(logger, current_priorities, current_alpha, step_count)
            
        print('\rEpisode {}\tAverage Score: {:.2f}\tLR: {:.6f}'.format(i_episode, np.mean(scores_window), current_lr), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tLR: {:.6f}'.format(i_episode, np.mean(scores_window), current_lr))
            torch.save(agent.qnetwork_local.state_dict(), Path(__file__).parent / 'checkpoint.pth')
            plt.plot(np.arange(len(scores)), scores)
            plt.savefig('p1_value_based_methods/reports/navigation_pixel_score.png')
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), Path(__file__).parent / 'checkpoint.pth')
            break

        close_logger(logger)

    # plot the scores
    plt.plot(np.arange(len(scores)), scores)
    plt.savefig('p1_value_based_methods/reports/navigation_pixel_score.png')
    #plt.show()


if __name__ == "__main__":
    main()