import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

def perform_pca_and_plot(
    pca_3d,
    pca_3d_last_layer,
    malicious_indices,
    DATASET,
    AGGREGATE_METHOD,
    ATTACT_TYPE,
    DP_TYPE,
    NUM_CLIENTS):
        
    # Convert list of epoch arrays into a single 3D NumPy array.
    # Final shape will be (num_epochs_saved, 101, d)
    pca_3d = np.stack(pca_3d, axis=0)
    print("Final 3D array shape for Full Model:", pca_3d.shape)
    pca_3d_last_layer = np.stack(pca_3d_last_layer, axis=0)
    print("Final 3D array shape for Last Layer:", pca_3d_last_layer.shape)
    
    # --- Creation of logs for the PCA results of all flatten models and last layers ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_log = os.path.join(script_dir, "results", "pca_logs")
    os.makedirs(save_path_log, exist_ok=True) 

    file_name_full_model = f"{DATASET}_{AGGREGATE_METHOD}_{ATTACT_TYPE}_{DP_TYPE}_Full_Model.log"
    file_name_last_layer = f"{DATASET}_{AGGREGATE_METHOD}_{ATTACT_TYPE}_{DP_TYPE}_Last_Layer.log"

    log_file_full_model = os.path.join(save_path_log, file_name_full_model)
    log_file_last_layer = os.path.join(save_path_log, file_name_last_layer)

    logger_pca_full_model = logging.getLogger("PCA_Logger_Full_Model")
    logger_pca_full_model.setLevel(logging.INFO)

    logger_pca_last_layer = logging.getLogger("PCA_Logger_Last_Layer")
    logger_pca_last_layer.setLevel(logging.INFO)

    if not logger_pca_full_model.handlers:
        file_handler_full_model = logging.FileHandler(log_file_full_model)
        file_handler_full_model.setFormatter(logging.Formatter("%(message)s"))
        logger_pca_full_model.addHandler(file_handler_full_model)
        
    if not logger_pca_last_layer.handlers:
        file_handler_last_layer = logging.FileHandler(log_file_last_layer)
        file_handler_last_layer.setFormatter(logging.Formatter("%(message)s"))
        logger_pca_last_layer.addHandler(file_handler_last_layer)
        
    # --- End of creation of log files ---

    # There are going to be C(NUM_CLIENTS) number of clients and one server
    labels = [f"C{i}" for i in range(NUM_CLIENTS)] + ["S"]
        
    # Save path for figures
    save_path = os.path.join(script_dir, "results", "pca_plots")
    os.makedirs(save_path, exist_ok=True)
    server_index = labels.index("S")  

    def get_color(index, malicious_indices, server_index):
        """ Returns the color for a given index: red for malicious, black for benign clients, blue for the server """
        if malicious_indices and index in malicious_indices:
            return 'red'  # Malicious
        elif index == server_index:
            return 'blue'  # Server
        else:
            return 'black'  # Benign

    # ========== FULL MODEL PCA & PLOTTING ==========
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns
    axs = axs.flatten()

    for epoch_idx, epoch_data in enumerate(pca_3d):
        epoch_data_clean = np.nan_to_num(epoch_data, nan=0.0)  # Replace NaN with 0
        pca_full_model = PCA(n_components=4)
        pca_result_full_model = pca_full_model.fit_transform(epoch_data_clean)

        # Log PCA result
        logger_pca_full_model.info("Epoch {} PCA result (n_components=4):".format(epoch_idx + 1))
        for label, row in zip(labels, pca_result_full_model):
            logger_pca_full_model.info("{}: {}".format(label, row))

        # Plot first 6 epochs
        if epoch_idx < 6:
            ax = axs[epoch_idx]
            colors = [get_color(i, malicious_indices, server_index)
                        for i in range(len(labels))]

            ax.scatter(pca_result_full_model[:, 0],
                        pca_result_full_model[:, 1],
                        c=colors, marker='o')

            for idx, label in enumerate(labels):
                ax.annotate(label,
                            (pca_result_full_model[idx, 0],
                            pca_result_full_model[idx, 1]),
                            fontsize=8,
                            alpha=0.7)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Epoch {epoch_idx + 1}: PC1 vs PC2")
            ax.grid(True)

    plt.suptitle(
        f"PC1 vs PC2 Across 6 Epochs Full Model - {AGGREGATE_METHOD} - {ATTACT_TYPE} - {DP_TYPE}",
        fontsize=20
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    file_name = (f"{DATASET}_{AGGREGATE_METHOD}_{ATTACT_TYPE}_"
                f"{DP_TYPE}_Full_Model_PC1_vs_PC2_Across_6_Epochs.png")
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== LAST LAYER PCA & PLOTTING ==========
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    for epoch_idx, epoch_data in enumerate(pca_3d_last_layer):
        epoch_data_clean = np.nan_to_num(epoch_data, nan=0.0)
        pca_last_layer = PCA(n_components=4)
        pca_result_last_layer = pca_last_layer.fit_transform(epoch_data_clean)

        # Log PCA result
        logger_pca_last_layer.info("Epoch {} Last Layer PCA result (n_components=4):".format(epoch_idx + 1))
        for label, row in zip(labels, pca_result_last_layer):
            logger_pca_last_layer.info("{}: {}".format(label, row))

        # Plot first 6 epochs
        if epoch_idx < 6:
            ax = axs[epoch_idx]
            colors = [get_color(i, malicious_indices, server_index)
                        for i in range(len(labels))]

            ax.scatter(pca_result_last_layer[:, 0],
                        pca_result_last_layer[:, 1],
                        c=colors, marker='o')

            for idx, label in enumerate(labels):
                ax.annotate(label,
                            (pca_result_last_layer[idx, 0],
                            pca_result_last_layer[idx, 1]),
                            fontsize=8,
                            alpha=0.7)

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Epoch {epoch_idx + 1}: PC1 vs PC2")
            ax.grid(True)

    plt.suptitle(
        f"PC1 vs PC2 Across 6 Epochs Last Layer - {AGGREGATE_METHOD} - {ATTACT_TYPE} - {DP_TYPE}",
        fontsize=20
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    file_name = (f"{DATASET}_{AGGREGATE_METHOD}_{ATTACT_TYPE}_"
                f"{DP_TYPE}_Last_Layer_PC1_vs_PC2_Across_6_Epochs.png")
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()

def process_epoch_updates(
    local_params,
    server,
    DP_TYPE,
    DATASET,
    epoch
):
    epoch_updates = []
    epoch_last_layer_updates = []

    # 1) Process local (client) updates
    for update in local_params:

        if isinstance(update, torch.Tensor):
            update = update.detach().cpu().numpy()

        # Flatten the entire model parameters
        update_flat = flatten_update(update)
        epoch_updates.append(update_flat)

        # Extract and flatten only the last layer
        last_layer_flat = extract_last_layer(update, DP_TYPE, DATASET)
        epoch_last_layer_updates.append(last_layer_flat)

    # 2) Flatten the server model update
    server_updates = []
    for key, value in server.model.state_dict().items():
        tensor_flat = value.view(-1).detach().cpu().numpy()
        server_updates.append(tensor_flat)
    server_flat = np.concatenate(server_updates)
    epoch_updates.append(server_flat)

    # 3) Extract the server’s last-layer parameters
    server_state = server.model.state_dict()
    if DP_TYPE == "No_DP" and DATASET == "CIFAR-10":
        last_layer_weight_server = server_state["model.fc.weight"]
        last_layer_bias_server = server_state["model.fc.bias"]
    elif DP_TYPE == "DP" and DATASET == "CIFAR-10":
        last_layer_weight_server = server_state["_module.model.fc.weight"]
        last_layer_bias_server = server_state["_module.model.fc.bias"]
    else:
        # For non-CIFAR-10 or other DP conditions
        if DP_TYPE == "No_DP":
            last_layer_weight_server = server_state["fc2.weight"]
            last_layer_bias_server = server_state["fc2.bias"]
        else:
            last_layer_weight_server = server_state["_module.fc2.weight"]
            last_layer_bias_server = server_state["_module.fc2.bias"]

    last_layer_flat_server = np.concatenate([
        last_layer_weight_server.view(-1).detach().cpu().numpy(),
        last_layer_bias_server.view(-1).detach().cpu().numpy()
    ])
    epoch_last_layer_updates.append(last_layer_flat_server)

    # 4) Stack client+server full-model updates for this epoch
    epoch_array = np.stack(epoch_updates, axis=0)  # shape: (num_clients+1, d)
    print(f"Epoch {epoch}: Stored epoch array with shape {epoch_array.shape}")

    # 5) Stack client+server last-layer updates for this epoch
    epoch_last_layer_array = np.stack(epoch_last_layer_updates, axis=0)  # shape: (num_clients+1, last_layer_dim)
    print(f"Epoch {epoch}: Stored last-layer array with shape {epoch_last_layer_array.shape}")
    return epoch_array, epoch_last_layer_array

def flatten_update(update):
    """
    Flattens an update that can be either a tensor or an OrderedDict of tensors.
    Returns a 1D NumPy array.
    """

    if isinstance(update, torch.Tensor):
        return update.view(-1).detach().cpu().numpy()
    
    elif isinstance(update, dict): 
        flat_list = []
        for key, tensor in update.items():
            flat_list.append(tensor.view(-1).detach().cpu().numpy())
        return np.concatenate(flat_list)
    
    else:
        raise TypeError("Unsupported type for flatten_update")
    
def extract_last_layer(update, DP_TYPE, dataset):
    prefix = "_module." if DP_TYPE != "No_DP" else ""

    if dataset == "CIFAR-10":
        weight_key = f"{prefix}model.fc.weight"
        bias_key = f"{prefix}model.fc.bias"
    else:
        weight_key = f"{prefix}fc2.weight"
        bias_key = f"{prefix}fc2.bias"

    if weight_key not in update or bias_key not in update:
        raise KeyError(f"Expected keys '{weight_key}' and/or '{bias_key}' not found in update.")

    last_layer_weight_flat = update[weight_key].view(-1).detach().cpu().numpy()
    last_layer_bias_flat = update[bias_key].view(-1).detach().cpu().numpy()

    return np.concatenate([last_layer_weight_flat, last_layer_bias_flat])


def perform_pca_no_plot_single_epoch(
    pca_3d,
    pca_3d_last_layer,
    malicious_indices,
    DATASET,
    AGGREGATE_METHOD,
    ATTACT_TYPE,
    DP_TYPE,
    NUM_CLIENTS,
    epoch_number
):
    import os
    import logging
    from sklearn.decomposition import PCA
    import numpy as np

    epoch_data_full_model = np.nan_to_num(pca_3d[0], nan=0.0)
    epoch_data_last_layer = np.nan_to_num(pca_3d_last_layer[0], nan=0.0)

    # Create log directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_log = os.path.join(script_dir, "results", "pca_logs")
    os.makedirs(save_path_log, exist_ok=True)

    file_name_full_model = f"{DATASET}_{AGGREGATE_METHOD}_{ATTACT_TYPE}_{DP_TYPE}_Full_Model.log"
    file_name_last_layer = f"{DATASET}_{AGGREGATE_METHOD}_{ATTACT_TYPE}_{DP_TYPE}_Last_Layer.log"

    log_file_full_model = os.path.join(save_path_log, file_name_full_model)
    log_file_last_layer = os.path.join(save_path_log, file_name_last_layer)

    # Set up loggers
    logger_full = logging.getLogger("PCA_Single_Epoch_Full")
    logger_full.setLevel(logging.INFO)
    if not logger_full.handlers:
        fh = logging.FileHandler(log_file_full_model)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger_full.addHandler(fh)

    logger_last = logging.getLogger("PCA_Single_Epoch_Last")
    logger_last.setLevel(logging.INFO)
    if not logger_last.handlers:
        fh = logging.FileHandler(log_file_last_layer)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger_last.addHandler(fh)

    # Labels
    labels = [f"C{i}" for i in range(NUM_CLIENTS)] + ["S"]

    # PCA for full model
    pca_full = PCA(n_components=4)
    pca_result_full = pca_full.fit_transform(epoch_data_full_model)
    logger_full.info(f"Epoch {epoch_number} PCA result (n_components=4):")
    for label, vec in zip(labels, pca_result_full):
        logger_full.info(f"{label}: {vec}")

    # PCA for last layer
    pca_last = PCA(n_components=4)
    pca_result_last = pca_last.fit_transform(epoch_data_last_layer)
    logger_last.info(f"Epoch {epoch_number} Last Layer PCA result (n_components=4):")
    for label, vec in zip(labels, pca_result_last):
        logger_last.info(f"{label}: {vec}")
