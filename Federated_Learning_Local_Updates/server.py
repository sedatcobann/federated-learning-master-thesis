import torch
import numpy as np
from tqdm import tqdm
import copy
from scipy.optimize import minimize
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
import opacus
from opacus.validators import ModuleValidator
import logging.config
from model import CNN, ResNet18_CIFAR10
import os
import csv

logger = None  

class Server:
    def __init__(self, model_type, input_channels, flatten_size, dp_type, aggregation_method=None, attack_type=None, dataset=None):
        self.input_channels = input_channels # Needed for Model Creation
        self.flatten_size = flatten_size # Needed for Model Creation
        self.dp_type = dp_type
        self.model = model_type(input_channels=input_channels, flatten_size = flatten_size) #Creates the model
        if self.dp_type == "DP": # If DP is chosen, prepare the server's model for privacy. This one does not clip weights or add noise to weights. It will use same naming for state_dicts as clients
            self.model = ModuleValidator.fix(self.model)
            self.privacy_engine = opacus.PrivacyEngine()
            self.model = self.privacy_engine._prepare_model(self.model)
        self.model.train()
        self.global_params = self.model.state_dict()
        self.momentum_vector = None # Used for Clustering Aggregation method
        self.learning_rate_clustering = None # Used for Clustering Aggregation method
        self.previous_norms = None # Used for Clipped Clustering Aggregation method
        self.logger = logger

        # Initialize logging for Identified Malicious Clients
        if self.logger is None:
            try:
                self.logger = logging.getLogger("MaliciousClientLogger") 
                self.logger.setLevel(logging.DEBUG)

                file_handler = logging.FileHandler(f"results/malicious_client_logs/{dataset}_{aggregation_method}_MalClient_{attack_type}_{dp_type}.log")
                console_handler = logging.StreamHandler()

                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)

                logging.getLogger("MalClientLogger").setLevel(logging.ERROR)  
                self.logger.info("Logging setup complete for MaliciousClientLogger.")
            except Exception as e:
                print(f"Error initializing logging: {e}")

    # Aggregation methods: FedAvg, Krum, MKrum, TrimmedMean, Median, GeoMedian, CenteredClipping, Clustering, ClippedClustering, CosDefense, Bulyan 
    def aggregate(self, local_params, aggregate_method, num_malicious_clients, client_data_sizes):
        if aggregate_method == "FedAvg":
            self.global_params = fedAverage(local_params, client_data_sizes)
        if aggregate_method == "Krum":
            self.global_params = krum(local_params, num_malicious_clients, self.logger)
        if aggregate_method == "MKrum":
            self.global_params = mkrum(local_params, num_malicious_clients, self.logger)
        if aggregate_method == "TrimmedMean":
            self.global_params = trimmed_mean(local_params, num_malicious_clients)
        if aggregate_method == "Median":
            self.global_params = median_aggregation(local_params)
        if aggregate_method == "GeoMedian":
            self.global_params = geometric_median_aggregation(local_params)
        if aggregate_method == "CenteredClipping":
            self.global_params = centered_clipping(local_params, copy.deepcopy(self.global_params), 1)
        if aggregate_method == "Clustering":
            if self.learning_rate_clustering == None:
                self.learning_rate_clustering = 1.0
            self.global_params, self.momentum_vector, self.learning_rate_clustering = clustering_aggregation(local_updates=local_params, learning_rate= self.learning_rate_clustering, momentum_vector=self.momentum_vector, global_update=copy.deepcopy(self.global_params), logger=self.logger)
        if aggregate_method == "ClippedClustering":
            self.global_params = clipped_clustering(local_params, max_tau=1e5, global_params=copy.deepcopy(self.global_params), logger = self.logger)
        if aggregate_method == "CosDefense":
            self.global_params = cos_defense(local_params, copy.deepcopy(self.global_params), self.logger)
        if aggregate_method == "Bulyan":
            self.global_params = bulyan(local_params, num_malicious_clients, self.logger)
            
        self.model.load_state_dict(self.global_params, strict=False)

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))

    def test_model(self, data, batch_size):
        return test(self.model, data, batch_size)

#FedAvg
def fedAverage(local_params, client_data_sizes):
    # Start with a copy of the first client's state_dict
    device = next(iter(local_params[0].values())).device
    total_samples = sum(client_data_sizes)
    avg_params = {key: torch.zeros_like(value).float().to(device) for key, value in local_params[0].items()}

    num_clients = len(local_params)
    # Sum up trainable parameters
    for i in range(0, num_clients):
        weight = client_data_sizes[i] / total_samples
        for key in avg_params.keys():
            avg_params[key] += local_params[i][key].float().to(device) * weight

    return avg_params

# Krum method
def krum(local_params, num_malicious_clients, logger):
    num_clients = len(local_params)
    num_neighbors = num_clients - num_malicious_clients - 2  # Number of closest neighbors to consider

    distances = torch.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 0.0
            for key in local_params[i]:
                param_i = local_params[i][key]
                param_j = local_params[j][key]

                if param_i.dtype in [torch.float32, torch.float64, torch.float16]:
                    dist += ((param_i - param_j).norm()) ** 2
                else:
                    continue

            distances[i, j] = distances[j, i] = dist

    scores = []
    for i in range(num_clients):
        sorted_distances, _ = torch.sort(distances[i])
        score = sorted_distances[1:num_neighbors + 1].sum()
        scores.append(score)

    # --- CSV Logging Section: Each row = 1 epoch ---
    # Get directory of current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define path for the CSV file in the same directory
    csv_path = os.path.join(script_dir, 'krum_cifar_non_iid_datap_dp_scores.csv')
    if csv_path is not None:
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write header only once
            if not file_exists:
                header = [f'client_{i}' for i in range(num_clients)]
                writer.writerow(header)
            writer.writerow([float(score) if isinstance(score, torch.Tensor) else score for score in scores])
    krum_index = torch.argmin(torch.tensor(scores)).item()

    if logger is not None:
        logger.warning(f"{krum_index}")
    return local_params[krum_index]

# MKrum method
def mkrum(local_params, num_malicious_clients, logger):
    num_clients = len(local_params)
    m_max = num_clients - num_malicious_clients - 2
    m_min = num_malicious_clients
    m = int((m_max + m_min)/2)
    num_neighbors = num_clients - num_malicious_clients - 2  # Number of closest neighbors to consider

    # Step 1: Calculate pairwise Euclidean distances between client updates
    distances = torch.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 0.0
            for key in local_params[i]:
                param_i = local_params[i][key]
                param_j = local_params[j][key]
                
                if param_i.dtype in [torch.float32, torch.float64, torch.float16]:
                    dist += ((param_i - param_j).norm()) ** 2
                else:
                    continue

            distances[i, j] = distances[j, i] = dist

    # Step 2: Score each client's update by summing distances to its closest neighbors
    scores = []
    for i in range(num_clients):
        # Get the sorted distances from client i to all other clients, excluding itself
        sorted_distances, _ = torch.sort(distances[i])
        score = sorted_distances[1:num_neighbors + 1].sum()  # Sum of distances to closest neighbors
        scores.append(score)

    # --- CSV Logging Section: Each row = 1 epoch ---
    # Get directory of current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define path for the CSV file in the same directory
    csv_path = os.path.join(script_dir, 'mkrum_cifar10_non_iid_no_attack_dp_scores.csv')
    if csv_path is not None:
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write header only once
            if not file_exists:
                header = [f'client_{i}' for i in range(num_clients)]
                writer.writerow(header)
            writer.writerow([float(score) if isinstance(score, torch.Tensor) else score for score in scores])

    # Step 3: Select the top-m clients with the smallest scores
    scores_tensor = torch.tensor(scores)
    top_m_indices = torch.argsort(scores_tensor)[:m]  # Indices of top-m clients

    # Identify the malicious clients (those not in top-m indices)
    all_indices = set(range(num_clients))
    top_m_indices_set = set(top_m_indices.tolist())
    malicious_client_indices = all_indices - top_m_indices_set
    #top_m_indices = [1,2,6,7,11,13,14,15,19] #No-LDP chosen clients
    #top_m_indices = [2,6,7,8,9,10,12,13,14,15,17,18,19] # No-LDP chosen clients + 4 clients

    print("Malicious Client Indices:", malicious_client_indices)
    if logger is not None:
        logger.warning(f"{list(malicious_client_indices)}")

    # Step 4: Average the parameters of the selected clients
    aggregated_params = {key: torch.zeros_like(local_params[0][key]) for key in local_params[0]}
    for idx in top_m_indices:
        for key in local_params[idx]:
            aggregated_params[key] += local_params[idx][key]
    for key in aggregated_params:
        if aggregated_params[key].dtype not in [torch.float32, torch.float64, torch.float16]:
            aggregated_params[key] = aggregated_params[key].float()
        aggregated_params[key] /= m


    return aggregated_params

def trimmed_mean(local_params, num_malicious_clients):
    num_clients = len(local_params)
    
    trim_fraction = min(0.2, num_malicious_clients / num_clients)
    num_trim = min(int(num_clients * trim_fraction), (num_clients - 1) // 2)  

    aggregated_params = {}

    for key in local_params[0]:
        param_dtype = local_params[0][key].dtype
        if not param_dtype.is_floating_point:
            aggregated_params[key] = local_params[0][key]
            continue

        param_values = torch.stack([local_params[i][key] for i in range(num_clients)])
        sorted_values, _ = torch.sort(param_values, dim=0)

        if num_trim >= num_clients // 2:
            trimmed_values = sorted_values
        else:
            trimmed_values = sorted_values[num_trim : num_clients - num_trim]

        aggregated_params[key] = trimmed_values.mean(dim=0)

    return aggregated_params

# Median method
def median_aggregation(local_params):
    num_clients = len(local_params)
    first_key = next(iter(local_params[0].keys()))
    device = local_params[0][first_key].device

    # Step 1: Initialize the aggregated parameters
    aggregated_params = {key: torch.zeros_like(local_params[0][key]) for key in local_params[0]}

    global_wins = torch.zeros(num_clients, dtype=torch.long, device=device)

    for key in local_params[0]:
        stacked = torch.stack([p[key] for p in local_params], dim=0)

        med_vals, med_idxs = torch.median(stacked, dim=0)

        # store the aggregated params
        aggregated_params[key] = med_vals

        # count wins per client for this layer
        counts = torch.bincount(med_idxs.flatten(), minlength=num_clients)

        global_wins += counts
  
    wins_list = [int(w) for w in global_wins.cpu().tolist()]
    log_path = "coordinate_wins_median_ldp_non_iid_datap.csv"
    # prepare CSV
    write_header = not os.path.exists(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            # header row: client_0, client_1, … 
            header = [f"client_{i}" for i in range(num_clients)]
            writer.writerow(header)
        # just the wins; row order = epoch order
        writer.writerow(wins_list)
    # after all layers, log the total wins per client
    for client_idx, wins in enumerate(global_wins.tolist()):
        print(f"client_{client_idx:>2d} has {wins:>4d} winning coordinates")

    return aggregated_params

# Helper for geometric median aggregation method to approximate geometric median
def weiszfeld_algorithm(points, tol=1e-5, max_iter=100):
#points (np.ndarray): Array of shape (n_points, dim).
#tol (float): Convergence tolerance.
#max_iter (int): Maximum number of iterations.

    median = np.mean(points, axis=0)  # Start with the arithmetic mean
    for _ in range(max_iter):
        distances = np.linalg.norm(points - median, axis=1)
        non_zero_distances = np.maximum(distances, tol) 
        weights = 1 / non_zero_distances
        new_median = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - median) < tol:
            return new_median
        median = new_median
    return median

def geometric_median_aggregation(local_params, tol=1e-5, max_iter=100):
#tol (float): Convergence tolerance for the Weiszfeld algorithm.
#max_iter (int): Maximum number of iterations for the Weiszfeld algorithmc.

    # Extract keys from the first parameter set
    param_keys = local_params[0].keys()

    batch_size = 5 
    # Initialize the aggregated parameters
    aggregated_params = {key: 0.0 for key in param_keys}
    
    for key in param_keys:
        # Collect all values of the current parameter from all clients
        values = np.array([params[key].detach().cpu().numpy() for params in local_params])
        
        # Flatten the values for batching
        flattened_values = values.reshape(values.shape[0], -1)
        
        # Create batches and compute batch means
        n_clients = flattened_values.shape[0]
        n_batches = n_clients // batch_size
        batch_means = []
        
        # Calculate batch_means
        for i in range(n_batches):
            batch = flattened_values[i * batch_size:(i + 1) * batch_size]
            batch_mean = np.mean(batch, axis=0)
            batch_means.append(batch_mean)
        
        # Handle remaining clients if they don't fit in complete batches
        if n_clients % batch_size != 0:
            remaining = flattened_values[n_batches * batch_size:]
            batch_means.append(np.mean(remaining, axis=0))
        
        batch_means = np.array(batch_means)

        # Compute the geometric median of the batch means
        median = weiszfeld_algorithm(batch_means, tol=tol, max_iter=max_iter)
        
        # Reshape back to the original parameter shape
        aggregated_params[key] = median.reshape(values.shape[1:])
        aggregated_params = {key: value.clone().detach() if isinstance(value, torch.Tensor) else torch.tensor(value) for key, value in aggregated_params.items()}

    return aggregated_params


# Centered Clipping method
def centered_clipping(local_params, global_params, tau=1):
 
    global_np = ordered_dict_to_numpy(global_params)
    
    local_np_list = [ordered_dict_to_numpy(params) for params in local_params]
    
    clipped_gradients = []
    for params_np in local_np_list:
        # Calculate the difference between the local update and the global model
        diff = params_np - global_np
        
        # Compute the norm of the difference
        norm = np.linalg.norm(diff)
        
        # Clip scaling
        scale = min(1, tau / norm) if norm != 0 else 1
        diff = diff*scale
        # Add the clipped difference to clipped gradients
        clipped_gradients.append(diff)
    
    # Compute the average of the clipped gradients
    average_clipped_gradients = np.mean(clipped_gradients, axis=0)
    average_clipped_gradients_global = global_np + average_clipped_gradients
    updated_global_params = numpy_to_ordered_dict(average_clipped_gradients_global, global_params)
    
    return updated_global_params

def clustering_aggregation(local_updates, clustering_threshold=0.02, learning_rate=1.0, momentum=0.5, momentum_vector=None, global_update=None, logger=None):
#clustering_threshold (float): Threshold for clustering similarity.
#learning_rate (float): Learning rate for adaptive update adjustment.
#momentum (float): Momentum factor for temporal adjustment.
#momentum_vector (numpy array): Previous momentum vector for temporal adjustment.
    os.makedirs('cosine_logs', exist_ok=True)
    log_file_path = 'cosine_logs/cifar_10_non_iid_dp_datap_clustering_cosine_similarity_log.csv'

    def flatten_update(update):
        if isinstance(update, dict):
            return np.concatenate([v.detach().cpu().numpy().flatten() for v in update.values()])
        return update

    def reconstruct_update(flattened_update, template):
        reconstructed = OrderedDict()
        current_index = 0
        for key, value in template.items():
            shape = value.shape
            size = int(np.prod(shape))
            reconstructed[key] = torch.tensor(flattened_update[current_index:current_index + size].reshape(shape))
            current_index += size
        return reconstructed

    template_update = local_updates[0]  # 
    local_updates = [flatten_update(update) for update in local_updates]

    # Step 1: Compute pairwise cosine similarity matrix
    num_clients = len(local_updates)
    similarity_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                similarity_matrix[i, j] = 1 - cosine(local_updates[i], local_updates[j])

    tri_indices = np.triu_indices(len(similarity_matrix), k=1)
    flattened_distances = similarity_matrix[tri_indices]
    with open(log_file_path, 'a') as f:
        line = ",".join(f"{val:.6f}" for val in flattened_distances) + "\n"
        f.write(line)

    # Step 2: Perform agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='complete')
    labels = clustering.fit_predict(1 - similarity_matrix)  # Convert similarity to distance

    # Step 3-4: Compare cluster similarity with threshold
    s_c1_c2 = np.max(similarity_matrix[labels == 0][:, labels == 1])
    cluster_sizes = [np.sum(labels == i) for i in np.unique(labels)]
    # Identify the cluster with fewer members (potentially malicious)
    malicious_cluster_label = np.argmin(cluster_sizes)  # Smallest cluster
    malicious_indices = [i for i in range(num_clients) if labels[i] == malicious_cluster_label]

    if s_c1_c2 < clustering_threshold:
        # Select updates from the larger cluster
        benign_cluster_label = np.argmax(cluster_sizes)
        benign_updates = [local_updates[i] for i in range(num_clients) if labels[i] == benign_cluster_label]
    else:
        # Treat all clients as benign if clusters are too dissimilar
        benign_updates = local_updates
    if logger is not None:
        logger.warning(f"{malicious_indices}")
    print(len(benign_updates))

    # Step 4: Aggregate benign updates by median
    benign_updates = np.array(benign_updates)
    aggregated_update = np.median(benign_updates, axis=0)

    flattened_global_update = flatten_update(global_update)

    # Step 5: Temporal adjustment using momentum
    delta_w = flattened_global_update - aggregated_update
    if momentum_vector is None:
        momentum_vector = delta_w
    else:
        momentum_vector = momentum * momentum_vector + (1 - momentum) * delta_w

    # Step 6: Adaptive learning rate adjustment
    cosine_similarity = 1 - cosine(delta_w, momentum_vector)
    if cosine_similarity <= 0:
        return global_update, momentum_vector, learning_rate 
    else:
        learning_rate = learning_rate * cosine_similarity
        learning_rate = np.clip(learning_rate, 1.0, 1.6)  # Clamp within range
        adjusted_update = flattened_global_update - learning_rate * momentum_vector

    global_update = reconstruct_update(adjusted_update, template_update)

    return global_update, momentum_vector, learning_rate

def centered_clipping_helper(local_params, global_params, previous_norms=None, default_tau=1.0):
#previous_norms (list of float, optional): List of previous norms for calculating the median.
#default_tau (float): Default threshold to use if previous norms are not provided.

    # Calculate the clipping threshold
    if previous_norms:
        tau = np.median(previous_norms)
    else:
        tau = default_tau

    tau = min(tau, default_tau)
    local_np_list = [ordered_dict_to_numpy(params) for params in local_params]

    clipped_gradients = []
    current_norms = []  # Store the norms of the current round

    for params_np in local_np_list:
        # Compute the norm of the update
        norm = np.linalg.norm(params_np)
        current_norms.append(norm)  # Add the norm to the current round's norms

        # Clip the update using the formula from the screenshot
        scaling_factor = min(1, tau / norm) if norm > 0 else 1
        clipped_update = params_np * scaling_factor

        clipped_gradients.append(clipped_update)

    clipped_updates = [numpy_to_ordered_dict(grad, global_params) for grad in clipped_gradients]

    return clipped_updates, current_norms

def cos_defense(local_updates, global_params, logger):
    os.makedirs('cosine_logs', exist_ok=True)
    log_file_path = 'cosine_logs/cifar_10_non_iid_dp_datap_cosdefense_cosine_similarity_log.csv'

    #cosine-similarity func
    def cosine_similarity(vec1, vec2):
        device = vec1.device 
        vec2 = vec2.to(device) 
        return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + 1e-8)
    
    # Extract last layer weights from global parameters
    global_last_layer = list(global_params.values())[-1].view(-1)
    
    # Compute cosine similarity for each client's last layer
    cosine_scores = []
    for update in local_updates:
        local_last_layer = list(update.values())[-1].view(-1)
        score = torch.abs(cosine_similarity(global_last_layer, local_last_layer))
        cosine_scores.append(score.item())
    
    with open(log_file_path, 'a') as f:
        line = ",".join(f"{val:.6f}" for val in cosine_scores) + "\n"
        f.write(line)

    # Normalize cosine similarity scores (min-max normalization)
    scores_tensor = torch.tensor(cosine_scores)
    normalized_scores = (scores_tensor - scores_tensor.min()) / (scores_tensor.max() - scores_tensor.min() + 1e-8)
    
    # Compute threshold as the mean of normalized scores
    threshold = normalized_scores.mean().item()
    
    # Identify benign updates (cosine similarity > threshold)
    benign_updates = []
    benign_indices = []

    for index, (update, score) in enumerate(zip(local_updates, normalized_scores)):
        if score > threshold:
            benign_updates.append(update)
            benign_indices.append(index)

    # Print the indices of the chosen updates
    malicious_indices = [i for i in range(len(local_updates)) if i not in benign_indices]
    if logger is not None:
        logger.warning(f"{malicious_indices}")

    # If no benign updates, return global parameters
    if not benign_updates:
        return copy.deepcopy(global_params)

    aggregated_update = OrderedDict()
    for key in global_params.keys():
        if key not in benign_updates[0]:  
            aggregated_update[key] = global_params[key]
        elif "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            aggregated_update[key] = benign_updates[0].get(key, global_params[key])
        else:
            aggregated_update[key] = torch.stack(
                [update[key] for update in benign_updates if key in update]
            ).mean(dim=0)
    
    return aggregated_update

def bulyan(local_updates, f, logger):
    reference_dict = local_updates[0]  

    n = len(local_updates)
    if n <= 4 * f + 3:
        raise ValueError("Number of clients must be greater than 4 * f + 3")

    original_indices = list(range(len(local_updates)))

    remaining_updates = local_updates.copy()  # Avoid modifying the original list
    remaining_indices = original_indices.copy()  # Track indices of remaining updates

    # Step 1: Use Krum to select (n - 2f) updates
    selected_updates = []
    for _ in range(n - 2 * f):
        selected_update = krum(remaining_updates, f, None)  # Pass state_dict objects to Krum
        selected_updates.append(ordered_dict_to_numpy(selected_update))  
        # Identify the index of the selected update
        selected_idx = next(i for i, update in enumerate(remaining_updates) if id(update) == id(selected_update))

        # Remove the selected update from remaining_updates and update indices
        del remaining_updates[selected_idx]
        del remaining_indices[selected_idx]  # Keep track of the remaining indices

    # Log the indices of remaining updates
    if logger is not None:
        logger.warning(f"{remaining_indices}")
    
    # Step 2: Compute Bulyan aggregation
    stacked_updates = np.vstack(selected_updates)
    bulyan_update = []
    theta = len(selected_updates)  # Total number of selected updates
    beta = theta - 2 * f  # Number of coordinates to average closest to the median

    for i in range(stacked_updates.shape[1]):
        column = stacked_updates[:, i]
        # Compute the median of the i-th coordinates
        median = np.median(column)
        
        # Compute the absolute differences to the median
        distances_to_median = np.abs(column - median)
        
        # Find the indices of the beta closest values to the median
        closest_indices = np.argsort(distances_to_median)[:beta]
        
        # Average the beta closest values
        closest_values = column[closest_indices]
        bulyan_update.append(np.mean(closest_values))

    bulyan_update = np.array(bulyan_update)

    return numpy_to_ordered_dict(bulyan_update, reference_dict)

def clipped_clustering(local_updates, max_tau=1e5, global_params=None, logger=None):
#max_tau (float): Maximum threshold for clipping norm.
    os.makedirs('cosine_logs', exist_ok=True)
    log_file_path = 'cosine_logs/cifar_10_non_iid_dp_datap_clippedclustering_cosine_similarity_log.csv'

    def flatten_ordered_dict(ordered_dict: OrderedDict) -> torch.Tensor:
        """Flatten an OrderedDict into a 1D tensor."""
        return torch.cat([v.flatten() for v in ordered_dict.values()])
    
    def reconstruct_ordered_dict(flattened_tensor: torch.Tensor, template: OrderedDict) -> OrderedDict:
        """Reconstruct an OrderedDict from a flattened tensor using a template OrderedDict."""
        reconstructed = OrderedDict()
        current_index = 0
        for key, value in template.items():
            numel = value.numel()
            shape = value.shape
            reconstructed[key] = flattened_tensor[current_index : current_index + numel].view(shape)
            current_index += numel
        return reconstructed

    flattened_inputs = [flatten_ordered_dict(update) if isinstance(update, OrderedDict) else update for update in local_updates]

    # Step 1: Clip updates based on L2 norm
    l2norm_his = [torch.norm(update).item() for update in flattened_inputs]
    threshold = np.median(l2norm_his)
    threshold = min(threshold, max_tau)

    clipped_updates = []
    for idx, l2 in enumerate(l2norm_his):
        if l2 > threshold:
            clipped_updates.append(flattened_inputs[idx] * (threshold / l2))
        else:
            clipped_updates.append(flattened_inputs[idx])

    clipped_updates = torch.stack(clipped_updates, dim=0)

    # Step 2: Compute pairwise cosine similarity
    num = len(clipped_updates)
    dis_max = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                clipped_updates[i, :], clipped_updates[j, :], dim=0
            )
            dis_max[j, i] = dis_max[i, j]
    dis_max[np.isinf(dis_max)] = 2
    dis_max[np.isnan(dis_max)] = 2

    tri_indices = np.triu_indices(len(dis_max), k=1)
    flattened_distances = dis_max[tri_indices]
    with open(log_file_path, 'a') as f:
        line = ",".join(f"{val:.6f}" for val in flattened_distances) + "\n"
        f.write(line)

    # Step 3: Cluster updates using AgglomerativeClustering
    clustering = AgglomerativeClustering(
        metric="precomputed", linkage="average", n_clusters=2
    )
    clustering.fit(dis_max)

    # Select the cluster with the majority of updates
    flag = 1 if np.sum(clustering.labels_) > len(dis_max) // 2 else 0
    selected_idxs = [
        idx for idx, label in enumerate(clustering.labels_) if label == flag
    ]

    all_indices = set(range(num))
    malicious_client_indices = all_indices - set(selected_idxs)
    if logger is not None:
        logger.warning(f"{list(malicious_client_indices)}")

    # Step 4: Aggregate selected updates
    benign_updates = [clipped_updates[idx] for idx in selected_idxs]
    aggregated_tensor = torch.mean(torch.stack(benign_updates), dim=0)

    if global_params is None:
        raise ValueError("global_params (template OrderedDict) must be provided.")
    aggregated_params = reconstruct_ordered_dict(aggregated_tensor, global_params)

    return aggregated_params

def test(model, dataloader, BATCH_SIZE):
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    model.eval()

    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset)/BATCH_SIZE):
        data, target = data, target
        device = next(model.parameters()).device  
        data = data.to(device) 
        target = target.to(device)
        output = model(data) 
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    
    return test_loss/len(dataloader.dataset), preds, accuracy

def strip_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_module."):
            new_key = key[len("_module."):]  # Remove '_module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

#Add '_module.' prefix to state_dict keys to match GradSampleModule format.
def add_module_prefix(state_dict):
    return {f"_module.{key}": value for key, value in state_dict.items()}

# Helper for CenteredClipping. Converts OrderedDict to a NumPy array.
def ordered_dict_to_numpy(ordered_dict):
    return np.concatenate([param.cpu().flatten().numpy() for param in ordered_dict.values()])

# Helper for CenteredClipping.
def numpy_to_ordered_dict(np_array, reference_dict):
    new_dict = OrderedDict()
    current_index = 0
    for key, value in reference_dict.items():
        num_elements = value.numel()
        # Convert numpy array slice back to torch.Tensor
        new_dict[key] = torch.tensor(
            np_array[current_index:current_index + num_elements].reshape(value.shape),
            dtype=value.dtype
        )
        current_index += num_elements
    return new_dict

