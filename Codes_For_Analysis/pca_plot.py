import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.lines import Line2D
#Used to create PCA plots from logs. It takes the whole folder
def plot_pca_from_logfile_styled(log_file, save_folder=None, NUM_CLIENTS=20, malicious_indices=None):
    if malicious_indices is None:
        malicious_indices = []

    base_name = os.path.basename(log_file).replace(".log", "")
    parts = base_name.split("_")

    if len(parts) < 5:
        raise ValueError("Expected format: <dataset>_<method>_<attack>_<dp>_<model_type>.log")

    dataset = parts[0].upper()
    method = parts[1]

    attack_raw = parts[2]
    if parts[3] == "No" and parts[4] == "DP":
        dp_raw = "No_DP"
        model_raw = "_".join(parts[5:])
    else:
        dp_raw = parts[3]
        model_raw = "_".join(parts[4:])

    attack_map = {
        "ModelP": "Model Poisoning",
        "DataP": "Data Poisoning",
        "No_Attack": "No Attack"
    }

    dp_map = {
        "DP": "DP",
        "No_DP": "No DP"
    }

    model_map = {
        "Full_Model": "Full Model",
        "Last_Layer": "Last Layer"
    }

    attack = attack_map.get(attack_raw, attack_raw)
    dp = dp_map.get(dp_raw, dp_raw)
    model_type = model_map.get(model_raw, model_raw.replace("_", " "))
    print(attack)
    if attack=="No":
        malicious_indices = []

    plot_title = f"PC1 vs PC2 Across 6 Epochs {model_type} - {method} - {attack} - {dp}"
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = os.path.join(save_folder, f"{base_name}_PC1_vs_PC2_Across_6_Epochs.png")

    if not save_file_name:
        save_base = base_name + "_PC1_vs_PC2_Across_6_Epochs.png"
        save_file_name = os.path.join("results", "pca_plots_from_logs", save_base)

    epochs_data = []
    current_epoch_data = []

    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Epoch"):
                if current_epoch_data:
                    epochs_data.append(np.array(current_epoch_data))
                    current_epoch_data = []
            elif line:
                try:
                    _, vector_part = line.split(":", 1)
                    vector = np.fromstring(vector_part.strip().strip("[]"), sep=' ')
                    current_epoch_data.append(vector)
                except ValueError:
                    print(f"Skipping line: {line}")
        if current_epoch_data:
            epochs_data.append(np.array(current_epoch_data))

    labels = [f"{i}" for i in range(NUM_CLIENTS)] + ["S"]
    server_index = NUM_CLIENTS

    def get_color(index):
        if index in malicious_indices:
            return "red"
        elif index == server_index:
            return "blue"
        else:
            return "black"

    # --- Plot first 6 epochs ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    for epoch_idx in range(min(6, len(epochs_data))):
        epoch_data = epochs_data[epoch_idx]
        ax = axs[epoch_idx]

        # Step 1: Plot normal clients (black circles)
        for i in range(NUM_CLIENTS):
            if i not in malicious_indices:
                ax.scatter(epoch_data[i, 0], epoch_data[i, 1],
                        color="black", marker='o', s=40, zorder=1)
                ax.annotate(labels[i], (epoch_data[i, 0], epoch_data[i, 1]),
                            fontsize=8, color='black', alpha=0.7, zorder=2)

        # Step 2: Plot malicious clients (red circles, transparent and bigger)
        for i in malicious_indices:
            if i < len(epoch_data):
                ax.scatter(epoch_data[i, 0], epoch_data[i, 1],
                        color='red', marker='o', s=70, alpha=0.7, zorder=3)
                ax.annotate(labels[i], (epoch_data[i, 0], epoch_data[i, 1]),
                            fontsize=8, color='black', alpha=0.7, zorder=4)

        # Step 3: Plot server (blue X) on top
        sx, sy = epoch_data[server_index, 0], epoch_data[server_index, 1]
        ax.scatter(sx, sy, color='blue', marker='X', s=100, zorder=5)
        ax.annotate("S", (sx, sy), fontsize=10, color='blue', fontweight='bold', zorder=6)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Epoch {epoch_idx + 1}: PC1 vs PC2")
        ax.grid(True)

    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label='Benign Client',
            markerfacecolor='black', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='o', color='red', label='Malicious Client',
            markerfacecolor='red', alpha=0.7, markersize=8, linestyle='None'),
        Line2D([0], [0], marker='X', color='blue', label='Server',
            markerfacecolor='blue', markersize=9, linestyle='None')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    plt.suptitle(plot_title, fontsize=20)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
    print(save_file_name)
    plt.close()


log_files = glob.glob(os.path.join("", "*.log")) # Folder path of the pca logs
if not log_files:
    print(f"No log files found in: {log_files}")

print(f"Found {len(log_files)} log files in '{log_files}'. Processing...")

for log_file in log_files:
    try:
        print(f"Processing: {os.path.basename(log_file)}")
        plot_pca_from_logfile_styled(
            log_file=log_file,
            save_folder="", # Folder to save the plots
            NUM_CLIENTS=20, # Number of clients
            malicious_indices=[0,1,2,3,7,8,11,16] # Malicious clients
        )
    except Exception as e:
        print(f"❌ Failed to process {log_file}: {e}")