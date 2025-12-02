import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# malicious_indices = [3, 4, 11, 13, 14, 17, 27, 28, 29, 31, 35, 54, 64, 69, 71, 75, 77, 81, 86, 94] ##MNIST

malicious_indices = [0, 8, 3, 7] ##CIFAR-10
# Load logs
dp_log = pd.read_csv('', header=None).values
no_dp_log = pd.read_csv('', header=None).values
num_clients = dp_log.shape[1]
benign_indices = [i for i in range(num_clients) if i not in malicious_indices]
epochs = np.arange(1, dp_log.shape[0] + 1)

# Containers for DP
dp_thresholds = []
dp_mal_means = []
dp_mal_stds = []
dp_above_thresh_means = []
dp_above_thresh_stds = []
dp_red_dots = []

for epoch_idx, epoch_data in enumerate(dp_log):
    min_val, max_val = epoch_data.min(), epoch_data.max()
    norm = (epoch_data - min_val) / (max_val - min_val + 1e-8)
    threshold = np.mean(norm)

    mal_scores = norm[malicious_indices]
    dp_thresholds.append(threshold)
    dp_mal_means.append(np.mean(mal_scores))
    dp_mal_stds.append(np.std(mal_scores))

    selected_scores = norm[norm > threshold]
    if len(selected_scores) > 0:
        dp_above_thresh_means.append(np.mean(selected_scores))
        dp_above_thresh_stds.append(np.std(selected_scores))
    else:
        dp_above_thresh_means.append(np.nan)
        dp_above_thresh_stds.append(0.0)

    for score in mal_scores:
        if score > threshold:
            dp_red_dots.append((epoch_idx + 1, score))

# Containers for No-DP
no_dp_thresholds = []
no_dp_mal_means = []
no_dp_mal_stds = []
no_dp_benign_means = []
no_dp_benign_stds = []

for epoch_data in no_dp_log:
    min_val, max_val = epoch_data.min(), epoch_data.max()
    norm = (epoch_data - min_val) / (max_val - min_val + 1e-8)
    threshold = np.mean(norm)

    mal_scores = norm[malicious_indices]
    benign_scores = norm[benign_indices]

    no_dp_thresholds.append(threshold)
    no_dp_mal_means.append(np.mean(mal_scores))
    no_dp_mal_stds.append(np.std(mal_scores))
    no_dp_benign_means.append(np.mean(benign_scores))
    no_dp_benign_stds.append(np.std(benign_scores))

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# DP Plot
axes[0].set_title("With Local Differential Privacy (LDP)")
if dp_red_dots:
    x_pts, y_pts = zip(*dp_red_dots)
    axes[0].scatter(x_pts, y_pts, color='red', alpha=0.6, label='Malicious > Threshold')

axes[0].plot(epochs, dp_mal_means, color='blue', linewidth=2, label='Mean Malicious Score')

axes[0].plot(epochs, dp_above_thresh_means, color='green', linewidth=2, label='Mean of Aggregated Clients > Threshold')


axes[0].plot(epochs, dp_thresholds, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Normalized Cosine Similarity")
axes[0].grid(True)

# No-DP Plot
axes[1].set_title("Without Differential Privacy (No-LDP)")
axes[1].plot(epochs, no_dp_mal_means, color='blue', linewidth=2, label='Mean Malicious Score')

axes[1].plot(epochs, no_dp_benign_means, color='green', linewidth=2, label='Mean Benign Score')


axes[1].plot(epochs, no_dp_thresholds, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

# Final layout
axes[0].legend()
axes[1].legend()
fig.suptitle("CosDefense: Threshold and Client Behavior (LDP vs No-LDP)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
