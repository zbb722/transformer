import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "transformer"  # 和训练时保持一致

def extract_scalars(log_dir, tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        print(f"Tag '{tag}' not found in logs.")
        return [], []
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_and_save(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Log directory {LOG_DIR} does not exist.")
        return

    # 训练每 epoch loss
    steps, train_epoch_loss = extract_scalars(LOG_DIR, "Train/Epoch_Loss")
    if steps:
        plot_and_save(steps, train_epoch_loss, "Epoch", "Loss", "Train Epoch Loss", "train_epoch_loss.png")

    # 验证每 epoch loss
    steps, valid_epoch_loss = extract_scalars(LOG_DIR, "Valid/Epoch_Loss")
    if steps:
        plot_and_save(steps, valid_epoch_loss, "Epoch", "Loss", "Valid Epoch Loss", "valid_epoch_loss.png")

    # 验证 BLEU
    steps, valid_bleu = extract_scalars(LOG_DIR, "Valid/BLEU")
    if steps:
        plot_and_save(steps, valid_bleu, "Epoch", "BLEU Score", "Valid BLEU Score", "valid_bleu.png")

    # 训练批次 loss (通常会很多点，画图时可考虑只画部分或加点标记)
    steps, train_batch_loss = extract_scalars(LOG_DIR, "Train/Batch_Loss")
    if steps:
        plt.figure(figsize=(12, 5))
        plt.plot(steps, train_batch_loss, linewidth=0.5)
        plt.xlabel("Batch Step")
        plt.ylabel("Loss")
        plt.title("Train Batch Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("train_batch_loss.png")
        plt.close()
        print("Saved plot: train_batch_loss.png")

if __name__ == "__main__":
    main()
