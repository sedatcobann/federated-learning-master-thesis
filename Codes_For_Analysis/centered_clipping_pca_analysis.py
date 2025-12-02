import numpy as np
import matplotlib.pyplot as plt
import re

# Paths to log files
dp_log_path = ''
no_dp_log_path = ''

def parse_pca_log(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    epoch_data = {}
    current_epoch = None
    current_vectors = []

    for line in lines:
        if line.startswith("Epoch"):
            if current_epoch is not None and current_vectors:
                epoch_data[current_epoch] = np.vstack(current_vectors)
            current_epoch = int(re.findall(r'\d+', line)[0])
            current_vectors = []
        elif line.startswith("C") or line.startswith("S"):
            numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line)
            numbers = list(map(float, numbers))

            if len(numbers) == 4:
                current_vectors.append(numbers)
            else:
                print(f"Warning: Skipped a line with {len(numbers)} numbers in epoch {current_epoch}")

    if current_epoch is not None and current_vectors:
        epoch_data[current_epoch] = np.vstack(current_vectors)

    return epoch_data



def compute_metrics(epoch_data):
    pca_variances = []
    mean_distances_to_server = []
    
    for epoch, vectors in sorted(epoch_data.items()):
        clients = vectors[:-1]  
        server = vectors[-1]
        
        variances = np.var(clients, axis=0)  
        print(variances)
        avg_variance = np.mean(variances)
        pca_variances.append(avg_variance)
        
        distances = np.linalg.norm(clients - server, axis=1)
        mean_distance = np.mean(distances)
        mean_distances_to_server.append(mean_distance)
    
    return pca_variances, mean_distances_to_server

# Parse both logs
dp_data = parse_pca_log(dp_log_path)
no_dp_data = parse_pca_log(no_dp_log_path)

# Compute metrics
dp_variances, dp_mean_distances = compute_metrics(dp_data)
no_dp_variances, no_dp_mean_distances = compute_metrics(no_dp_data)

# Plotting
dp_epochs = range(1, len(dp_variances) + 1)
no_dp_epochs = range(1, len(no_dp_variances) + 1)
min_len = min(len(dp_variances), len(no_dp_variances))
dp_epochs = range(1, min_len + 1)
no_dp_epochs = range(1, min_len + 1)

dp_variances = dp_variances[:min_len]
no_dp_variances = no_dp_variances[:min_len]
dp_mean_distances = dp_mean_distances[:min_len]
no_dp_mean_distances = no_dp_mean_distances[:min_len]
plt.figure(figsize=(14, 6))

# Variance plot
plt.subplot(1, 2, 1)
plt.plot(no_dp_epochs, no_dp_variances, label="No LDP", marker='o')
plt.plot(dp_epochs, dp_variances, label="With LDP", marker='x')
plt.title("Mean PCA Variance Across Clients")
plt.xlabel("Epoch")
plt.ylabel("Mean Variance")
plt.legend()
plt.grid(True)
plt.yscale('log')

# Mean Distance to Server plot
plt.subplot(1, 2, 2)
plt.plot(no_dp_epochs, no_dp_mean_distances, label="No LDP", marker='o')
plt.plot(dp_epochs, dp_mean_distances, label="With LDP", marker='x')
plt.title("Mean Client-to-Server Distance in PCA Space")
plt.xlabel("Epoch")
plt.ylabel("Mean Distance")
plt.legend()
plt.grid(True)
plt.suptitle("Centered Clipping: PCA Variance & Client-Server Distance (MNIST - IID_TYPE - ATTACK_TYPE)", fontsize=16)
plt.tight_layout()
plt.show()
