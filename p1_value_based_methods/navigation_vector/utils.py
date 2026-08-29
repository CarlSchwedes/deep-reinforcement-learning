# utils.py
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def init_logger(path, run_name="banana_dqn"):
    """Initialisiert den TensorBoard SummaryWriter mit einem Zeitstempel."""
    log_dir = f"{path}/{run_name}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return SummaryWriter(log_dir)

def log_step_metrics(writer, loss, avg_q, step_count):
    """Loggt Metriken, die bei jedem Lernschritt (Update) anfallen."""
    if loss is not None:
        writer.add_scalar('Train/Loss', loss, step_count)
    if avg_q is not None:
        writer.add_scalar('Train/Avg_Q_Value', avg_q, step_count)

def log_episode_metrics(writer, score, eps, episode_count):
    """Loggt Metriken am Ende jeder Episode."""
    writer.add_scalar('Reward/Episode_Score', score, episode_count)
    writer.add_scalar('Hyperparameters/Epsilon', eps, episode_count)

def close_logger(writer):
    """Schließt den Writer sauber, um Datenverlust zu vermeiden."""
    writer.close()

def log_per_distribution_plot(writer, priorities, alpha, step_count):
    """Generates a distribution plot of the PER buffer and logs it to TensorBoard."""
    actual_size = len(priorities)
    if actual_size == 0:
        return

    # 1. Compute probabilities for the current alpha and a uniform comparison
    probs_current = (priorities ** alpha)
    probs_current /= probs_current.sum()
    
    probs_uniform = np.ones_like(priorities) / actual_size

    # Sort from highest probability to lowest for a clean plot layout
    probs_current_sorted = np.sort(probs_current)[::-1]

    # 2. Build the matplotlib figure safely in memory
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(probs_current_sorted, label=f"Current PER (α={alpha})", color="orange", linewidth=2)
    ax.plot([0, actual_size], [probs_uniform[0], probs_uniform[0]], label="Uniform Replay (α=0)", color="red", linestyle="--")
    
    ax.set_title("PER Sampling Distribution Profile")
    ax.set_xlabel("Buffer Items (Sorted Highest to Lowest)")
    ax.set_ylabel("Sampling Probability")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    # 3. Convert matplotlib figure buffer to a PyTorch/TensorBoard compatible image tensor
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_rgba = np.asarray(rgba_buffer)
    
    # Extract RGB channels and transpose to (Channels, Height, Width) format for PyTorch
    image_rgb = image_rgba[:, :, :3]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)

    # 4. Log the image tensor to TensorBoard
    writer.add_image('PER/Sampling_Distribution', image_tensor, step_count)
    
    # Close the figure to free up system memory leakages
    plt.close(fig)
