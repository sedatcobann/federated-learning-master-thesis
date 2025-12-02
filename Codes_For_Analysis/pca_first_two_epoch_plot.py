import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.lines import Line2D

def compare_ldp_vs_no_ldp(logfile_no_dp, logfile_dp, save_folder=None, NUM_CLIENTS=20, malicious_indices=None):
    if malicious_indices is None:
        malicious_indices = []

    def load_pca_data(logfile):
        data = []
        current_epoch = []
        with open(logfile, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Epoch"):
                    if current_epoch:
                        data.append(np.array(current_epoch))
                        current_epoch = []
                elif line:
                    try:
                        _, vector_part = line.split(":", 1)
                        vector = np.fromstring(vector_part.strip().strip("[]"), sep=' ')
                        current_epoch.append(vector)
                    except ValueError:
                        print(f"Skipping line: {line}")
            if current_epoch:
                data.append(np.array(current_epoch))
        return data

    # Load both logs
    data_no_dp = load_pca_data(logfile_no_dp)
    data_dp = load_pca_data(logfile_dp)

    labels = [f"{i}" for i in range(NUM_CLIENTS)] + ["S"]
    server_index = NUM_CLIENTS

    def get_color(index):
        if index in malicious_indices:
            return "red"
        elif index == server_index:
            return "blue"
        else:
            return "black"

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 
    axs = axs.flatten()

    titles = ["Without LDP - Epoch 1", "With LDP - Epoch 1", "Without LDP - Epoch 2", "With LDP - Epoch 2"]
    epochs = [0, 0, 1, 1]
    datas = [data_no_dp, data_dp, data_no_dp, data_dp]

    for i, ax in enumerate(axs):
        epoch_data = datas[i][epochs[i]]

        for j in range(NUM_CLIENTS):
            if j not in malicious_indices:
                ax.scatter(epoch_data[j, 0], epoch_data[j, 1], color='black', marker='o', s=40)

        for j in malicious_indices:
            if j < len(epoch_data):
                ax.scatter(epoch_data[j, 0], epoch_data[j, 1], color='red', marker='o', s=70, alpha=0.7)

        sx, sy = epoch_data[server_index, 0], epoch_data[server_index, 1]
        ax.scatter(sx, sy, color='blue', marker='X', s=100, alpha=0.5)
        ax.annotate("S", (sx, sy), fontsize=10, color='blue', fontweight='bold')

        ax.set_title(titles[i])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
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
    plt.suptitle("PCA Comparison of First Two Epochs - With vs Without LDP (METHOD_NAME - DATASET - ATTACK_TYPE - IID_TYPE)", fontsize=18)
    plt.subplots_adjust(hspace=0.3)  
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    # if save_folder:
    #     os.makedirs(save_folder, exist_ok=True)
    #     base_name = os.path.basename(logfile_no_dp).replace(".log", "")
    #     save_path = os.path.join(save_folder, f"{base_name}_DP_vs_NoDP_Comparison_Epochs1_2.png")
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

compare_ldp_vs_no_ldp(
            logfile_no_dp = "", #No-LDP PCA logs
            logfile_dp="", #LDP PCA logs
            save_folder="",
            NUM_CLIENTS=20, #Number of clients
            malicious_indices=[0, 8, 3, 7] #Malicious clients
        )