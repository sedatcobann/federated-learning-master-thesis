import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Configuration
actual_malicious = set([3, 4, 11, 13, 14, 17, 27, 28, 29, 31,
                        35, 54, 64, 69, 71, 75, 77, 81, 86, 94])
num_clients = 100
all_clients = set(range(num_clients))

# Load cosine distance logs
dp_log = pd.read_csv('', header=None).values
no_dp_log = pd.read_csv('', header=None).values

# Load detected malicious client logs
with open('') as f:
    dp_mal_lines = f.readlines()

with open('') as f:
    no_dp_mal_lines = f.readlines()

# Parse detected malicious indices from logs
def parse_detected_malicious(lines):
    detected_per_epoch = []
    for line in lines:
        match = re.findall(r'\[(.*?)\]', line)
        if match:
            ids = [int(i.strip()) for i in match[0].split(',') if i.strip().isdigit()]
            detected_per_epoch.append(set(ids))
    return detected_per_epoch

dp_detected_malicious = parse_detected_malicious(dp_mal_lines)
no_dp_detected_malicious = parse_detected_malicious(no_dp_mal_lines)

# Reconstruct full cosine similarity matrix
def reconstruct_similarity(flattened, num_clients=100):
    matrix = np.zeros((num_clients, num_clients))
    upper = np.triu_indices(num_clients, k=1)
    matrix[upper] = flattened
    matrix += matrix.T
    np.fill_diagonal(matrix, 1)
    return 1 - matrix  # similarity = 1 - distance

# Main analysis function
def analyze_logs_with_std(log_data, detected_malicious_all_epochs):
    benign_means, benign_stds = [], []
    malicious_means, malicious_stds = [], []
    red_dots = []

    for epoch_idx, (flattened, detected_malicious) in enumerate(zip(log_data, detected_malicious_all_epochs)):
        sim_matrix = reconstruct_similarity(flattened)
        print(detected_malicious)
        benign_cluster = list(all_clients - detected_malicious)
        malicious_cluster = list(detected_malicious)

        if len(benign_cluster) > 1:
            benign_sims = [
                np.mean([sim_matrix[i][j] for j in benign_cluster if i != j]) 
                for i in benign_cluster
            ]
        else:
            benign_sims = [0]        
    
        # For malicious cluster
        if len(malicious_cluster) > 1:
            malicious_sims = [
                np.mean([sim_matrix[i][j] for j in malicious_cluster if i != j]) 
                for i in malicious_cluster
            ]
        elif len(malicious_cluster) == 1:
            malicious_sims = [0]  
        else:
            malicious_sims = [np.nan]  
            
        benign_means.append(np.mean(benign_sims) if benign_sims else 2)
        benign_stds.append(np.std(benign_sims) if benign_sims else 2)

        malicious_means.append(np.mean(malicious_sims) if malicious_sims else 2)
        malicious_stds.append(np.std(malicious_sims) if malicious_sims else 2)

        for i in actual_malicious:
            if i in benign_cluster:
                for j in benign_cluster:
                    if i != j:
                        red_dots.append((epoch_idx + 1, sim_matrix[i][j]))

    return benign_means, benign_stds, malicious_means, malicious_stds, red_dots

# Analyze
dp_ben_mean, dp_ben_std, dp_mal_mean, dp_mal_std, dp_red_dots = analyze_logs_with_std(dp_log, dp_detected_malicious)
no_dp_ben_mean, no_dp_ben_std, no_dp_mal_mean, no_dp_mal_std, _ = analyze_logs_with_std(no_dp_log, no_dp_detected_malicious)

min_len = min(len(dp_ben_mean), len(no_dp_ben_mean))

# Plotting
epochs = np.arange(1, min_len + 1)
dp_ben_mean = dp_ben_mean[:min_len]
dp_ben_std = dp_ben_std[:min_len]
dp_mal_mean = dp_mal_mean[:min_len]
dp_mal_std = dp_mal_std[:min_len]
no_dp_ben_mean = no_dp_ben_mean[:min_len]
no_dp_ben_std = no_dp_ben_std[:min_len]
no_dp_mal_mean = no_dp_mal_mean[:min_len]
no_dp_mal_std = no_dp_mal_std[:min_len]
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# --- LDP Plot ---
axes[1].set_title("With Local Differential Privacy (LDP)")
axes[1].plot(epochs, dp_ben_mean, color='green', linewidth=2, label='Mean Benign Cluster Similarity')
axes[1].fill_between(epochs, np.array(dp_ben_mean) - np.array(dp_ben_std),
                     np.array(dp_ben_mean) + np.array(dp_ben_std), color='green', alpha=0.2)
axes[1].plot(epochs, dp_mal_mean, color='blue', linewidth=2, label='Mean Malicious Cluster Similarity')
axes[1].fill_between(epochs, np.array(dp_mal_mean) - np.array(dp_mal_std),
                     np.array(dp_mal_mean) + np.array(dp_mal_std), color='blue', alpha=0.2)
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
axes[0].set_ylabel("Mean Cosine Similarity")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

# Final
fig.suptitle("Clipped Clustering: Cosine Similarity Within Clusters (±1 Std, No-LDP vs LDP)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
