import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss(train_history, val_history):
    epochs = range(1, len(train_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_history, label="Training Loss")
    plt.plot(epochs, val_history, label= "Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create images folder if it doesn't exist
    Path("images").mkdir(parents=True, exist_ok=True)

    # save the plot
    plt.savefig("images/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    

def plot_learning_rate_comparison(results):
    fig, axes = plt.subplots(1,2,figsize=(12,5))

    for lr, history in results.items():
        axes[0].plot(history["train"], label=f"LR {lr}")
        axes[1].plot(history["validation"], label=f"LR {lr}")

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()