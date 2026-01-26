import re
import matplotlib.pyplot as plt

# Path to your loss.txt file
LOG_PATH = "D:\image project\loss.txt"

train_losses = []
val_losses = []
epochs = []

# Read and extract data
with open(LOG_PATH, "r") as f:
    for line in f:
        match = re.search(r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)", line)
        if match:
            epoch = int(match.group(1))
            train = float(match.group(2))
            val = float(match.group(3))

            epochs.append(epoch)
            train_losses.append(train)
            val_losses.append(val)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="training loss", color="blue", linewidth=2)
plt.plot(epochs, val_losses, label="validation loss", color="orange", linewidth=2)

plt.xlabel("epoch", fontsize=12)
plt.ylabel("loss", fontsize=12)
plt.title("Training vs Validation Loss", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig("train_vs_val_loss.png")
plt.show()

print("Saved graph as train_vs_val_loss.png")
