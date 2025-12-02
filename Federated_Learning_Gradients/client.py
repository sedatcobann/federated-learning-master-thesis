import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import opacus
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
import torchmetrics
from model import CNN, ResNet18_CIFAR10
import gc
from collections import Counter
import pandas as pd

class Client:
    def __init__(self, client_id, device, model_type, data_loader, epsilon, delta, max_norm, local_iteration, epochs, num_classes = 10,
                 dp_type = "No_DP", attack_type = "No_Attack", malicious = False, learning_rate=0.05, input_channels=1, flatten_size = 64*7*7, 
                 global_model = None, dataset_type = "mnist", mpbs=12, batch_size = 64):
        self.client_id = client_id
        self.device = device
        self.dataset_type = dataset_type
        self.model = model_type(input_channels=input_channels, flatten_size = flatten_size) # Creation of Model via model_type (Should be separate for each client)
        if dp_type == "DP":
            self.model = ModuleValidator.fix(self.model)

        if dp_type == "DP" and dataset_type == "CIFAR-10":
            self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate) #lr should be 0.1
        elif dp_type=="No_DP" and dataset_type == "CIFAR-10":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=4e-5) # lr should be 0.05
        else: # For MNIST and FashionMNIST
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9) # lr should be 0.05

        self.data_loader = data_loader
        self.epsilon = epsilon # Target Epsilon
        self.delta = delta # Target Delta
        self.max_norm = max_norm # Needed for the clipping in DP Case
        self.local_iteration = local_iteration
        self.epochs = epochs # Needed for the calculation of epsilon will be used in each epoch for the client
        self.num_classes = num_classes
        self.dp_type = dp_type # DP or No_DP
        self.total_epsilon = 0 # To track whether the target epsilon is exceeded or not
        self.attack_type = attack_type # No_Attack, ModelP, DataP
        self.malicious = malicious
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.mpbs = mpbs
        self.batch_size = batch_size
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes = num_classes).to(device)

        if dp_type == "DP":
            self.initialize_privacy_engine() #Initializes privacy engine for DP

        self.model.load_state_dict(global_model) # Set the global_model's state_dict to be sure that each client and the server has the same model at the beginning
        self.optimizer_state = None

    def initialize_privacy_engine(self):
            print(f"{self.client_id} initializes privacy engine")
            self.privacy_engine = opacus.PrivacyEngine()
            sample_rate = self.data_loader.batch_size / len(self.data_loader.dataset)
            alphas = [1 + x * 0.1 for x in range(1, 100)] + list(range(10, 1000, 10))
            noise_multiplier = get_noise_multiplier(
                        target_epsilon=self.epsilon,
                        target_delta=self.delta,
                        sample_rate=sample_rate,
                        epochs=self.epochs*self.local_iteration,
                        alphas = alphas
                    )
            
            self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=self.max_norm,
                poisson_sampling=False
            )
        
    def run(self, global_model_latest):
        if self.data_loader: # If client has data to train
            if self.dp_type == "DP": # If DP is chosen
                if self.dataset_type == "CIFAR-10":
                    param, loss, local_epsilon, grad = self.train_with_dp2()
                else:
                    param, loss, local_epsilon, grad = self.train_with_dp()
                self.total_epsilon = local_epsilon
                print(f"Client {self.client_id} | Total ε: {self.total_epsilon:.4f}")
                return param, loss, grad
            else: # If No_DP is chosen
                param, loss, grad = self.train_without_dp()
                return param, loss, grad
        else:
            return None, None, None

    # For IPM attack
    def attack(self, global_model, benign_mean_grads, std_dev, lr):
        malicious_grads = {
            name: -1 * std_dev * tensor
            for name, tensor in benign_mean_grads.items()
        }
        dot_val = 0.0
        for name in benign_mean_grads:
            dot_val += torch.sum(benign_mean_grads[name] * malicious_grads[name])

        return malicious_grads


    def train_without_dp(self):
        self.model = self.model.to(self.device)
        train_loss = 0.0
        self.model.train()
        old_params = copy.deepcopy(self.model.state_dict())
        for i in range(self.local_iteration):
            batch_loss = 0.0
            for batch_idx, (data, target) in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                # Data has shape [batch_size, 1, 28, 28] for MNIST and FashionMNIST
                if data.dim() > 4:
                    data = data.squeeze()
                elif data.dim() == 3:
                    data = data.unsqueeze(1)

                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                batch_loss += loss.item() * data.size(0)
                loss.backward()
                self.optimizer.step()

            train_loss += batch_loss / len(self.data_loader)
        new_params = self.model.state_dict()
        pseudo_grads = {}
        for key in old_params.keys():
            pseudo_grads[key] = old_params[key] - new_params[key]
        return self.model.state_dict(), train_loss / self.local_iteration, pseudo_grads

    def train_with_dp(self):
            self.model = self.model.to(self.device)
            self.model.train()
            train_loss = 0.0
            epoch_accuracies = []  # To track accuracy per epoch
            old_params = copy.deepcopy(self.model.state_dict())

            for epoch in range(self.local_iteration):
                batch_loss = 0.0
                batch_accuracies = []  # To track batch-wise accuracy
                with BatchMemoryManager(
                    data_loader=self.data_loader,
                    max_physical_batch_size=32,
                    optimizer=self.optimizer
                ) as memory_safe_loader:
                    for batch_idx, (data, target) in tqdm(enumerate(memory_safe_loader), total=len(memory_safe_loader)):
                        data, target = data.to(self.device), target.to(self.device) 
                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.loss_function(output, target)
                        loss.backward()
                        grads = {
                            name: param.grad.clone().detach()
                            for name, param in self.model.named_parameters()
                            if param.grad is not None
                        }
                        self.optimizer.step()

                        batch_loss += loss.item() * data.size(0)
                        batch_accuracy = (output.argmax(dim=1) == target).float().mean().item()
                        batch_accuracies.append(batch_accuracy)

                # Calculate and log epoch results
                epoch_loss = batch_loss / len(self.data_loader)

                epoch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
                epoch_accuracies.append(epoch_accuracy)
                train_loss += epoch_loss
                print(f"Client {self.client_id} | Epoch {epoch + 1}/{self.local_iteration} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")

            # Retrieve privacy budget (epsilon) for this client
            privacy_epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

            new_params = self.model.state_dict()
            pseudo_grads = {}
            for key in old_params.keys():
                pseudo_grads[key] = old_params[key] - new_params[key]
            print(f"Client {self.client_id} | Final Avg Accuracy: {avg_accuracy:.4f} | Total ε: {privacy_epsilon:.4f}")
            return self.model.state_dict(), train_loss / self.local_iteration, privacy_epsilon, pseudo_grads


    def train_with_dp2(self):
            self.model = self.model.to(self.device) 
            self.model.train() 
            train_loss = 0.0 
            epoch_accuracies = []  # To track accuracy per epoch 
            old_params = copy.deepcopy(self.model.state_dict())

            for epoch in range(self.local_iteration):
                batch_loss = 0.0  
                batch_accuracies = []  # To track batch-wise accuracy  
                with BatchMemoryManager(
                    data_loader=self.data_loader,
                    max_physical_batch_size=self.mpbs,
                    optimizer=self.optimizer
                ) as memory_safe_loader:
                    for batch_idx, (data, target) in tqdm(enumerate(memory_safe_loader), total=len(memory_safe_loader)): 
                        data, target = data.to(self.device), target.to(self.device)
                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.loss_function(output, target)
                        loss.backward()
                        self.optimizer.step()

                        batch_loss += loss.item() * data.size(0)
                        batch_accuracy = (output.argmax(dim=1) == target).float().mean().item()
                        batch_accuracies.append(batch_accuracy)

                        #Clear memory after processing each batch
                        del data, target, output
                        torch.cuda.empty_cache()

                # Calculate and log epoch results
                epoch_loss = batch_loss / len(self.data_loader) 
                epoch_accuracy = sum(batch_accuracies) / len(batch_accuracies) 
                epoch_accuracies.append(epoch_accuracy) 
                train_loss += epoch_loss 
                print(f"Client {self.client_id} | Epoch {epoch + 1}/{self.local_iteration} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}") 

            # Retrieve privacy budget (epsilon) for this client
            privacy_epsilon = self.privacy_engine.get_epsilon(self.delta) 
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies) 

            new_params = self.model.state_dict()
            pseudo_grads = {}
            for key in old_params.keys():
                pseudo_grads[key] = old_params[key] - new_params[key]

            del memory_safe_loader
            gc.collect()
            torch.cuda.empty_cache()
            self.model.to('cpu') 

            print(f"Client {self.client_id} | Final Avg Accuracy: {avg_accuracy:.4f} | Total ε: {privacy_epsilon:.4f}")
            return self.model.state_dict(), train_loss / self.local_iteration, privacy_epsilon, pseudo_grads


# Called from the train_federated.py to train each client.
def federated_training(clients, global_model_latest):
        local_params, local_losses, benign_clients_grads, client_grads = [None] * len(clients), [None] * len(clients), [], [None] * len(clients)
        for client in clients:
            if(client.attack_type!="ModelP"):
                param, loss, grad = client.run(global_model_latest) 
                if param is not None and loss is not None:
                    local_params[int(client.client_id.split("_")[1])]=param
                    local_losses[int(client.client_id.split("_")[1])]=loss
                    client_grads[int(client.client_id.split("_")[1])]=grad

                    benign_clients_grads.append(grad)
        
        mean_benign_grads = {}
        for layer in benign_clients_grads[0].keys():
            layer_grads = torch.stack([grads[layer] for grads in benign_clients_grads])
            mean_benign_grads[layer] = torch.mean(layer_grads.float(), dim=0)

        for client in clients:
            if(client.malicious and client.attack_type=="ModelP"):
                param, loss, grad = client.run(global_model_latest)
                grad = client.attack(global_model_latest, mean_benign_grads, 0.1, client.learning_rate)
                if param is not None and loss is not None and grad is not None:
                    reconstructed_params = {}
                    global_state = global_model_latest.state_dict()
                    for key in global_state.keys():
                        reconstructed_params[key] = global_state[key] - grad[key].to(global_state[key].device)
                    local_params[int(client.client_id.split("_")[1])]=reconstructed_params
                    local_losses[int(client.client_id.split("_")[1])]=loss
                    client_grads[int(client.client_id.split("_")[1])]=grad
        max_epsilon = max(client.total_epsilon for client in clients)
        return copy.deepcopy(local_params), local_losses, max_epsilon, copy.deepcopy(client_grads)

# Creates clients. Assign num_malicious as malicious client, and if their attack type is Data Poisoning, do label flipping
def initialize_clients(num_clients, train_data, alpha, num_classes, batch_size, flip_probability, label_map, device, model_type, epsilon,
                       delta, max_norm, local_iteration, epochs, dp_type, num_malicious=0, attack_type="No_Attack", input_channels=1, flatten_size=64*7*7, global_model = None, learning_rate = 0.05, dataset_type="mnist", mpbs=12):
    # Generate a random sample of client indices to be malicious
    malicious_indices = set(random.sample(range(num_clients), num_malicious))
    # Generate data_loaders for each client
    train_data_loaders, client_data_sizes = distribute_data_to_clients(train_data, alpha, num_clients, num_classes, batch_size, malicious_indices, flip_probability,label_map, attack_type)

    clients = []
    for i in range(num_clients):
        # Set the attack type based on whether the client is malicious
        client_attack_type = attack_type if i in malicious_indices else "No_Attack"

        # Initialize the client
        client = Client(
            client_id=f"client_{i}",
            device=device,
            model_type=model_type,
            data_loader=train_data_loaders[i],
            epsilon=epsilon,
            delta=delta,
            max_norm=max_norm,
            local_iteration=local_iteration,
            epochs=epochs,
            num_classes=num_classes,
            dp_type=dp_type,
            attack_type=client_attack_type,
            malicious=(i in malicious_indices),
            learning_rate = learning_rate,
            input_channels = input_channels,
            flatten_size = flatten_size,
            global_model = copy.deepcopy(global_model),
            dataset_type = dataset_type,
            mpbs = mpbs,
            batch_size = batch_size
        )
        clients.append(client)
    return clients, malicious_indices, client_data_sizes

# Distribute data to clients using Dirichlet distribution and return data loaders for each client. Returns DataLoader for each client
def distribute_data_to_clients(train_data, alpha, num_clients, num_classes, batch_size=64, 
                               malicious_client_indices=None, flip_probability=0.7, 
                               label_map=None, attack_type=None, log_filename="client_data_log.csv"):

    if malicious_client_indices is None:
        malicious_client_indices = []

    class_data = {}
    for data, target in train_data:
        for i in range(len(data)):
            label = target[i].item()
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((data[i], target[i]))

    client_datasets = [[] for _ in range(num_clients)]
    for label, items in class_data.items():
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(items)).astype(int)

        start = 0
        for client_idx, num_items in enumerate(proportions):
            client_datasets[client_idx].extend(items[start:start + num_items])
            start += num_items

    client_loaders = []
    client_dataset_sizes = []
    log_data = []

    print("\n=== Client Data Distribution Summary ===")
    for client_idx, client_data in enumerate(client_datasets):
        if client_data:
            data_tensor = torch.cat([x[0].unsqueeze(0) if x[0].dim() == 3 else x[0] for x in client_data if x[0] is not None])
            target_tensor = torch.cat([x[1].unsqueeze(0) for x in client_data if x[1] is not None])

            if client_idx in malicious_client_indices and attack_type == "DataP":
                print(f"Client {client_idx} is malicious. Performing label flipping with probability {flip_probability}.")
                target_tensor = flip_labels(target_tensor, num_classes, flip_probability, label_map)

            client_dataset = TensorDataset(data_tensor, target_tensor)
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            label_counts = Counter(target_tensor.tolist())
            total_samples = len(client_dataset)
            unique_labels = len(label_counts)
            num_batches = len(client_loader)

            print(f"\nClient {client_idx}:")
            print(f"  Total samples: {total_samples}")
            print(f"  Unique labels: {unique_labels}")
            print(f"  Label distribution: {dict(label_counts)}")
            print(f"  Number of batches: {num_batches}")

            # Append stats to log list
            log_entry = {
                "client_id": client_idx,
                "total_samples": total_samples,
                "unique_labels": unique_labels,
                "num_batches": num_batches,
                "is_malicious": client_idx in malicious_client_indices
            }
            for class_id in range(num_classes):
                log_entry[f"class_{class_id}_count"] = label_counts.get(class_id, 0)
            log_data.append(log_entry)

            client_loaders.append(client_loader)
            client_dataset_sizes.append(total_samples)

        else:
            print(f"\nClient {client_idx} received no data.")
            log_entry = {
                "client_id": client_idx,
                "total_samples": 0,
                "unique_labels": 0,
                "num_batches": 0,
                "is_malicious": client_idx in malicious_client_indices
            }
            for class_id in range(num_classes):
                log_entry[f"class_{class_id}_count"] = 0
            log_data.append(log_entry)

            client_loaders.append(None)
            client_dataset_sizes.append(0)

    # Save log to CSV
    df_log = pd.DataFrame(log_data)
    df_log.to_csv(log_filename, index=False)
    print(f"\n✅ Client distribution log saved to '{log_filename}'")

    return client_loaders, client_dataset_sizes

# Flips a specified percentage of labels to predefined incorrect ones
def flip_labels(target_tensor, num_classes, flip_probability, label_map):
    flipped_target = target_tensor.clone()
    num_samples = len(flipped_target)

    # Determine indices to flip based on the flip probability
    flip_indices = np.random.choice(num_samples, int(num_samples * flip_probability), replace=False)

    for idx in flip_indices:
        current_label = flipped_target[idx].item()
        if label_map and current_label in label_map:
            flipped_target[idx] = label_map[current_label]
        else:
            raise ValueError(f"Label {current_label} is not defined in the label_map.")

    return flipped_target



