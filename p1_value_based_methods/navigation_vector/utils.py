# utils.py
import datetime
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
