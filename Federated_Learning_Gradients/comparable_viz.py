import matplotlib.pyplot as plt

def parse_log_file(file_path):
    epochs = []
    train_losses = []
    test_losses = []
    test_accuracies = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Epoch:" in line and "Train Loss:" in line and "Test Loss:" in line and "Test Accuracy:" in line:
                try:
                    parts = line.strip().split(",")

                    # Extract the epoch number
                    epoch_part = parts[0].split(":")[1]
                    epoch = int(epoch_part.split("/")[0].strip())

                    # Extract the train loss
                    train_loss = parts[1].split(":")[1].strip()
                    train_loss = float(train_loss) if train_loss.lower() != 'nan' else None

                    # Extract the test loss
                    test_loss = parts[2].split(":")[1].strip()
                    test_loss = float(test_loss) if test_loss.lower() != 'nan' else None

                    # Extract the test accuracy
                    test_accuracy = parts[3].split(":")[1].strip()
                    test_accuracy = float(test_accuracy)

                    # Append values to their respective lists
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    test_accuracies.append(test_accuracy)

                except (IndexError, ValueError) as e:
                    print(f"Skipping malformed line: {line.strip()} | Error: {e}")

    return epochs, train_losses, test_losses, test_accuracies


# Compare DP methods with same aggregation method
def compare_dp_with_same_aggregation_method(file_paths, labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Epochs=10, NumClients=10, MalClient=2, BatchSize=5, Epsilon=5")
    for file_path, label in zip(file_paths, labels):
        epochs, train_losses, test_losses, test_accuracies = parse_log_file(file_path)
        
        axes[0].plot(epochs, test_accuracies, label=label)
        axes[1].plot(epochs, train_losses, label=label)
    
    # Set titles and labels for accuracy plot
    axes[0].set_title("Test Accuracy Comparison")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].legend()
    axes[0].grid(True)
    
    # Set titles and labels for loss plot
    axes[1].set_title("Train Loss Comparison")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Test Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Compare different aggregation methods with same dp
def compare_aggregation_methods_with_different_dp(file_paths, labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Epochs=80, NumClients=100, BatchSize=64, Aggregation=Centered Clipping")
    for file_path, label in zip(file_paths, labels):
        epochs, train_losses, test_losses, test_accuracies = parse_log_file(file_path)
        
        # Plot validation accuracy
        axes[0].plot(epochs, test_accuracies, label=label)
        # Plot validation loss
        axes[1].plot(epochs, train_losses, label=label)
    
    # Set titles and labels for accuracy plot
    axes[0].set_title("Test Accuracy Comparison")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].legend()
    axes[0].grid(True)
    
    # Set titles and labels for loss plot
    axes[1].set_title("Train Loss Comparison")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Test Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()

log_file_paths = []
labels = []

#compare_dp_with_same_aggregation_method(log_file_paths, labels)
compare_aggregation_methods_with_different_dp(log_file_paths, labels)
