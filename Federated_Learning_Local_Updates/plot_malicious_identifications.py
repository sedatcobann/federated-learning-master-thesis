import matplotlib.pyplot as plt

# List of actual malicious clients
actual_malicious_clients = []

actual_malicious_set = set(actual_malicious_clients)

# Path to the log file for 4 different cases
log_files = []

# Define custom subtitles for each subplot
subtitles = []

# Function to parse a log file and calculate percentages
def parse_log_and_calculate(log_file_path):
    epochs = []
    percentages = []

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            # Check if the line contains identified malicious clients
            if line.startswith("["):
                # Extract client IDs from the log
                identified_clients = eval(line.strip())
                identified_clients_set = {f"client_{id}" for id in identified_clients}

                # Calculate the intersection with the actual malicious clients
                correct_identifications = actual_malicious_set.intersection(identified_clients_set)

                # Calculate precision: correctly identified / total identified
                if len(identified_clients_set) > 0:
                    precision = (len(correct_identifications) / len(identified_clients_set)) * 100
                else:
                    precision = 0  # Handle edge case where no clients are identified

                # Append results for plotting
                epochs.append(len(epochs) + 1)
                percentages.append(precision)

    return epochs, percentages

# Subplot configuration
subplot_layout = (2, 2)  
fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], figsize=(12, 8))

# Loop through log files and plot each case with a unique subtitle
for i, log_file in enumerate(log_files):
    row, col = divmod(i, subplot_layout[1])  
    ax = axes[row][col] if subplot_layout != (1, 1) else axes  
    epochs, percentages = parse_log_and_calculate(log_file)

    ax.plot(epochs, percentages, marker='o', linestyle='-')
    ax.set_title(subtitles[i])  
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision (%)")
    ax.grid(True)
    ax.legend()

plt.suptitle("Precision of Malicious Client Identification Across Different Cases - Method Name", fontsize=16)
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("results/malicious_client_logs/malicious_clients_identification_precision.png")
plt.show()