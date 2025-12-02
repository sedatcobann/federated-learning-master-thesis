import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load score CSV files
dp_scores_path = ""
no_dp_scores_path = ""

df_dp = pd.read_csv(dp_scores_path)
df_no_dp = pd.read_csv(no_dp_scores_path)

# Define malicious and benign client IDs
malicious_ids = {0, 1, 2, 3, 7, 8, 11, 16} #IPM Mal Client Ids for CIFAR-10
#malicious_ids = {0, 8, 3, 7} # Other Attacks Mal Client Ids for CIFAR-10

benign_ids = set(range(20)) - malicious_ids

# Function to compute average MKrum score per epoch for each group
def compute_stats(df):
    malicious_cols = [f"client_{i}" for i in malicious_ids]
    benign_cols    = [f"client_{i}" for i in benign_ids]
    avg_mal = df[malicious_cols].mean(axis=1)
    std_mal = df[malicious_cols].std(axis=1)
    avg_ben = df[benign_cols].mean(axis=1)
    std_ben = df[benign_cols].std(axis=1)
    return avg_mal, std_mal, avg_ben, std_ben

# Calculate average scores
dp_avg_mal, dp_std_mal, dp_avg_ben, dp_std_ben = compute_stats(df_dp)
no_dp_avg_mal, no_dp_std_mal, no_dp_avg_ben, no_dp_std_ben = compute_stats(df_no_dp)

epochs = np.arange(len(dp_avg_mal))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

# Without LDP
axes[0].plot(epochs, no_dp_avg_mal, linestyle='--', marker='x', label='Malicious')
axes[0].fill_between(epochs,
                     no_dp_avg_mal - no_dp_std_mal,
                     no_dp_avg_mal + no_dp_std_mal,
                     alpha=0.2)
axes[0].plot(epochs, no_dp_avg_ben, linestyle='-', marker='x', label='Benign')
axes[0].fill_between(epochs,
                     no_dp_avg_ben - no_dp_std_ben,
                     no_dp_avg_ben + no_dp_std_ben,
                     alpha=0.2)
axes[0].set_title("Without LDP")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Average MKrum Score")
axes[0].legend()
axes[0].grid(True)

# With LDP
axes[1].plot(epochs, dp_avg_mal, linestyle='--', marker='o', label='Malicious')
axes[1].fill_between(epochs,
                     dp_avg_mal - dp_std_mal,
                     dp_avg_mal + dp_std_mal,
                     alpha=0.2)
axes[1].plot(epochs, dp_avg_ben, linestyle='-', marker='o', label='Benign')
axes[1].fill_between(epochs,
                     dp_avg_ben - dp_std_ben,
                     dp_avg_ben + dp_std_ben,
                     alpha=0.2)
axes[1].set_title("With LDP")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

# Title and layout
fig.suptitle("Average MKrum Score per Epoch ±1 Std Dev\nDATASET – IID_TYPE – ATTACK_TYPE")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()