import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Configuration
actual_malicious = set([0,1,2,3,7,8,11,16]) # For Random Gaussian and LF Attack
num_clients = 20 # For MNIST, it is 100, for CIFAR-10, it is 20
all_clients = set(range(num_clients))

# Load cosine distance logs for clipped clustering
dp_log = pd.read_csv('', header=None).values
no_dp_log = pd.read_csv('', header=None).values

# Reconstruct full cosine similarity matrix
def reconstruct_similarity(flattened, num_clients=100):
    matrix = np.zeros((num_clients, num_clients))
    upper = np.triu_indices(num_clients, k=1)
    matrix[upper] = flattened
    matrix += matrix.T
    np.fill_diagonal(matrix, 1)
    return 1 - matrix  # similarity = 1 - distance

def analyze_logs_with_std(log_data):
    benign_means, benign_stds = [], []
    malicious_means, malicious_stds = [], []
    intercluster_means = []
    red_dots = []

    for epoch_idx, flattened in enumerate(log_data):
        sim_matrix = reconstruct_similarity(flattened, num_clients=num_clients)
        benign_cluster = list(all_clients - actual_malicious)
        malicious_cluster = list(actual_malicious)

        # Mean similarities
        if len(benign_cluster) > 1:
            benign_sims = [
                np.mean([sim_matrix[i][j] for j in benign_cluster if i != j]) 
                for i in benign_cluster
            ]
        else:
            benign_sims = [0]

        if len(malicious_cluster) > 1:
            malicious_sims = [
                np.mean([sim_matrix[i][j] for j in malicious_cluster if i != j]) 
                for i in malicious_cluster
            ]
        else:
            malicious_sims = [0]

        inter_sims = [
            sim_matrix[i][j]
            for i in benign_cluster
            for j in malicious_cluster
        ]
        intercluster_mean = np.mean(inter_sims) if inter_sims else 0
        print(intercluster_mean)

        benign_means.append(np.mean(benign_sims) if benign_sims else 2)
        benign_stds.append(np.std(benign_sims) if benign_sims else 2)

        malicious_means.append(np.mean(malicious_sims) if malicious_sims else 2)
        malicious_stds.append(np.std(malicious_sims) if malicious_sims else 2)

        intercluster_means.append(intercluster_mean)

        for i in actual_malicious:
            if i in benign_cluster:
                for j in benign_cluster:
                    if i != j:
                        red_dots.append((epoch_idx + 1, sim_matrix[i][j]))

    return benign_means, benign_stds, malicious_means, malicious_stds, intercluster_means, red_dots

# Analyze
dp_ben_mean, dp_ben_std, dp_mal_mean, dp_mal_std, dp_inter_mean, dp_red_dots = analyze_logs_with_std(dp_log)
no_dp_ben_mean, no_dp_ben_std, no_dp_mal_mean, no_dp_mal_std, no_dp_inter_mean, _ = analyze_logs_with_std(no_dp_log)

# Plotting
epochs = np.arange(1, len(dp_ben_mean) + 1)

min_len = min(len(dp_ben_mean), len(no_dp_ben_mean))

epochs = np.arange(1, min_len + 1)

# Truncate all metric lists to match the minimum length
dp_ben_mean = dp_ben_mean[:min_len]
dp_ben_std = dp_ben_std[:min_len]
dp_mal_mean = dp_mal_mean[:min_len]
dp_mal_std = dp_mal_std[:min_len]
no_dp_ben_mean = no_dp_ben_mean[:min_len]
no_dp_ben_std = no_dp_ben_std[:min_len]
no_dp_mal_mean = no_dp_mal_mean[:min_len]
no_dp_mal_std = no_dp_mal_std[:min_len]
dp_inter_mean = dp_inter_mean[:min_len]
no_dp_inter_mean = no_dp_inter_mean[:min_len]


fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# --- LDP Plot ---
axes[1].set_title("With Local Differential Privacy (LDP)")
axes[1].plot(epochs, dp_ben_mean, color='green', linewidth=2, label='Mean Benign Cluster Similarity')
axes[1].fill_between(epochs, np.array(dp_ben_mean) - np.array(dp_ben_std),
                     np.array(dp_ben_mean) + np.array(dp_ben_std), color='green', alpha=0.2)
axes[1].plot(epochs, dp_mal_mean, color='blue', linewidth=2, label='Mean Malicious Cluster Similarity')
axes[1].fill_between(epochs, np.array(dp_mal_mean) - np.array(dp_mal_std),
                     np.array(dp_mal_mean) + np.array(dp_mal_std), color='blue', alpha=0.2)
axes[1].plot(epochs, dp_inter_mean, color='orange', linewidth=2, linestyle='--', label='Mean Benign-Malicious Similarity')
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

# --- No-LDP Plot ---
axes[0].set_title("Without Differential Privacy (No-LDP)")
axes[0].plot(epochs, no_dp_ben_mean, color='green', linewidth=2, label='Mean Benign Cluster Similarity')
axes[0].fill_between(epochs, np.array(no_dp_ben_mean) - np.array(no_dp_ben_std),
                     np.array(no_dp_ben_mean) + np.array(no_dp_ben_std), color='green', alpha=0.2)
axes[0].plot(epochs, no_dp_mal_mean, color='blue', linewidth=2, label='Mean Malicious Cluster Similarity')
axes[0].fill_between(epochs, np.array(no_dp_mal_mean) - np.array(no_dp_mal_std),
                     np.array(no_dp_mal_mean) + np.array(no_dp_mal_std), color='blue', alpha=0.2)
axes[0].plot(epochs, no_dp_inter_mean, color='orange', linewidth=2, linestyle='--', label='Mean Benign-Malicious Similarity')

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Mean Cosine Similarity")
axes[0].legend()
axes[0].grid(True)

# Final
fig.suptitle("Clipped Clustering: Cosine Similarity Within Actual Benign and Malicious Clients (±1 Std, No-LDP vs LDP) - DATASET - IID_TYPE - ATTACK_TYPE", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
