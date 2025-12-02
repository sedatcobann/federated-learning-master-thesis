#!/usr/bin/python
# -*- coding: utf-8 -*-
# This is used for only IPM Attack. When you set ModelP as attack_type, it performs IPM Attack.
import os
import torch
import numpy as np
import logging.config
import random
from model import CNN, ResNet18_CIFAR10
from dataset import load_dataset, visualize_dataset
from client import initialize_clients, federated_training
from visualization import plot_training_progress
from server import Server
import argparse
import copy
import time 
from datetime import date
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pca import *

# Seeding
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.empty_cache()

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


MODEL_MAPPING = {
    "CNN": CNN,
    "ResNet18_CIFAR10": ResNet18_CIFAR10
    }

start_time = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Configuration")

    # Communication and local training settings
    parser.add_argument("--num_epochs", type=int, default=2, help="The number of communication rounds")
    parser.add_argument("--local_iters", type=int, default=1, help="The number of local epochs for clients")

    # Batch size
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")

    # Clients
    parser.add_argument("--num_clients", type=int, default=20, help="Total number of clients")

    #Learning Rate
    parser.add_argument("--lr", type=float, default=0.05, help="Learning Rate")

    # Data distribution
    parser.add_argument("--dirichlet_alpha", type=float, default=100, help="Alpha value for Dirichlet distribution (controls non-IID nature)")

    # Attack settings
    parser.add_argument("--attack_type", type=str, default="ModelP",
                        choices=["DataP", "ModelP", "No_Attack"], help="Type of attack to simulate")
    parser.add_argument("--percentage_of_malicious", type=float, default=40, help="Percentage of malicious clients (e.g., 20 for 20%)")
    parser.add_argument("--label_flip_per", type=float, default=0, help="Percentage of label flipping (e.g., 0.2 for 20%)")

    # Differential privacy settings
    parser.add_argument("--epsilon", type=float, default=8, help="Epsilon value for differential privacy")
    parser.add_argument("--delta", type=float, default=1e-3, help="Delta value for differential privacy")
    parser.add_argument("--dp_type", type=str, default="DP",
                        choices=["No_DP", "DP"], help="Type of differential privacy to apply")
    parser.add_argument("--max_norm", type=float, default=2.0, help="Maximum gradient norm for privacy engine")

    # Aggregation methods
    parser.add_argument("--aggregate_method", type=str, default="FedAvg",
                        choices=["FedAvg", "Krum", "MKrum", "TrimmedMean", "Median", "GeoMedian",
                                 "CenteredClipping", "Clustering", "ClippedClustering", "FLVS", "CosDefense", "Bulyan"],
                        help="Aggregation method for federated learning")

    # Model and dataset
    parser.add_argument("--model_type", type=str, default="ResNet18_CIFAR10",
                        choices=list(MODEL_MAPPING.keys()),
                        help="Model type to use")
    parser.add_argument("--dataset", type=str, default="CIFAR-10",
                        choices=["mnist", "fashion_mnist", "CIFAR-10"], help="Dataset to use")

    parser.add_argument("--mpbs", type=int, default=8, help="Maximum Physical Batch Size")

    # Parse arguments
    args = parser.parse_args()

    return args

args = parse_args()
today = date.today().isoformat()
logging.getLogger('matplotlib').setLevel(logging.ERROR)
NUM_EPOCHS = args.num_epochs # The number of communication round
LOCAL_ITERS = args.local_iters # The number of local epoch for clients
VIS_DATA = False # To visualize the first 5 of the dataset
BATCH_SIZE = args.batch_size # Batch size
NUM_CLIENTS = args.num_clients # The total number of clients
LR =  args.lr # Learning Rate
# DIRICHLET_ALPHA ~ 0.5, 0.1 --> non-IID
# DIRICHLET_ALPHA ~ 1 --> moderately non-IID
# DIRICHLET_ALPHA ~ >> 1 --> IID
DIRICHLET_ALPHA = args.dirichlet_alpha
ATTACT_TYPE =  args.attack_type # Attack Type: "DataP", "ModelP", "No_Attack"
PERCENTAGE_OF_MALICIOUS = args.percentage_of_malicious # Percentage of malicious client
NUM_MALICIOUS_CLIENT = int(NUM_CLIENTS*PERCENTAGE_OF_MALICIOUS/100) # The number of malicious client
LABEL_FLIP_PER = args.label_flip_per # Percentage of Label Flip (0.2 == 20%)
DP_TYPE = args.dp_type #Differential Privacy Type: "No_DP", "DP", ...
EPSILON, DELTA, MAX_NORM = (
    (0, 0, 0) if DP_TYPE == "No_DP" else (args.epsilon, args.delta, args.max_norm)
)
# Implemented aggregate methods: "FedAvg", "Krum", "MKrum", "TrimmedMean", "Median", GeoMedian, CenteredClipping, Clustering, ClippedClustering, CosDefense, Bulyan ...
AGGREGATE_METHOD= args.aggregate_method
MODEL_TYPE = MODEL_MAPPING[args.model_type] # Model Type: CNN, ResNet18_CIFAR10
# At the end of the training, if you want to see a plot for accuracy and loss changes through communication rounds, set it as True
LOSS_ACCURACY_VIZ = True
# At the end of the training, if you want to do PCA on the model and on the last layer of the model, set it as True
pca=True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET = args.dataset # DATASET can be "mnist", "fashion_mnist", "CIFAR-10"
MPBS = args.mpbs

# Used for label flipping
generic_label_map = {
    0: 9,  # Class 0 → Class 9
    1: 8,  # Class 1 → Class 8
    2: 7,  # Class 2 → Class 7
    3: 6,  # Class 3 → Class 6
    4: 5,  # Class 4 → Class 5
    5: 4,  # Class 5 → Class 4
    6: 3,  # Class 6 → Class 3
    7: 2,  # Class 7 → Class 2
    8: 1,  # Class 8 → Class 1
    9: 0   # Class 9 → Class 0
}

# Helper for calculating norms (used for CenteredClipping)
def compute_gradient_norms(model, dataloader, loss_fn, device):
    model.train()
    norms = []
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Compute the norm of the gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        norm = total_norm ** 0.5
        norms.append(norm)
        # Print the gradient norm for this batch
        print(f"Batch {batch_idx + 1}: Gradient Norm = {norm:.4f}")
    return norms

if __name__=="__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')
    
    print("Results directory exists or created:", os.path.isdir('results'))

    #Initialize a logger to log epoch results
    logname = ('results/loss_accuracy_logs/log_' + DATASET + "_CR_" + str(NUM_EPOCHS) 
            + "_C_" + str(NUM_CLIENTS) + "_MC_" + str(NUM_MALICIOUS_CLIENT) + "_AT_" + str(ATTACT_TYPE) + "_LI_" + str(LOCAL_ITERS) 
            + "_DA_" + str(DIRICHLET_ALPHA) + "_BS_" + str(BATCH_SIZE)
            + "_AGG_" + str(AGGREGATE_METHOD) + "_DP_" + str(DP_TYPE) + "_EPS_" + str(EPSILON) + "_LF_" + str(LABEL_FLIP_PER) + ".log")

    print("Log filename:", logname)

    try:
        logger = logging.getLogger("EpochLogger")
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(logname)
        console_handler = logging.StreamHandler()

        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.getLogger("opacus").setLevel(logging.ERROR) 
        logger.info("Logging setup complete for EpochLogger.")
    except Exception as e:
        print(f"Error initializing logging: {e}")

    # Create plot name for the graph.
    plot_name = ('results/loss_accuracy_plots/plot_federated_' + DATASET + "_EPOCHS:" + str(NUM_EPOCHS) 
               +"_CLIENTS:"+ str(NUM_CLIENTS) + "_MALCLIENTS:" + str(NUM_MALICIOUS_CLIENT) + "_AT:" + str(ATTACT_TYPE) + "_LOCAL_ITER:" + str(LOCAL_ITERS) 
               + "_DIRICHET_ALPHA:" + str(DIRICHLET_ALPHA) + "_BATCH_SIZE:" + str(BATCH_SIZE)
               + "_AGG:" + str(AGGREGATE_METHOD) + "_DP_TYPE:" + str(DP_TYPE) + "_EPS:" +str(EPSILON) + "_LF:" + str(LABEL_FLIP_PER))
    
    #get data
    train_data, test_data = load_dataset(batch_size=BATCH_SIZE, dataset=DATASET, dp_type = DP_TYPE)
    
    if VIS_DATA: visualize_dataset([train_data, test_data])

    # Input channel number and flatten size change with the dataset and model.
    input_channels, flatten_size = (3, 64*8*8) if DATASET == "CIFAR-10" else (1, 64*7*7) # CNN

    #creation of Server
    server = Server(MODEL_TYPE, input_channels, flatten_size, DP_TYPE, aggregation_method=AGGREGATE_METHOD, attack_type=ATTACT_TYPE, dataset=DATASET)

    # Uncomment this part to calculate norms of the gradients. (for centeredclipping)
    #criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    #norms = compute_gradient_norms(server.model,train_data, criterion, DEVICE)

    #Create Clients
    clients, malicious_indices, client_data_sizes = initialize_clients(num_clients=NUM_CLIENTS, train_data=train_data, alpha=DIRICHLET_ALPHA, num_classes=10, batch_size=BATCH_SIZE, 
                                 flip_probability= LABEL_FLIP_PER, label_map= generic_label_map, device = DEVICE, model_type= MODEL_TYPE, 
                                 epsilon=EPSILON, delta=DELTA, max_norm= MAX_NORM, local_iteration= LOCAL_ITERS, epochs= NUM_EPOCHS, 
                                 dp_type=DP_TYPE, num_malicious= int(NUM_MALICIOUS_CLIENT), attack_type = ATTACT_TYPE, input_channels = input_channels, 
                                 flatten_size = flatten_size, global_model=server.model.state_dict(), learning_rate=LR, dataset_type = DATASET, mpbs = MPBS)   

    all_train_loss, all_test_loss, all_accuracy = [], [], []
    val_loss_min = np.Inf

    pca_3d = [] # To use in the PCA Part
    pca_3d_last_layer = [] # To use in the PCA Part

    # Train the model for given number of epochs
    for epoch in range(1, NUM_EPOCHS + 1):
        print("\nEpoch:", str(epoch))
        local_params, local_losses, client_grads = [], [], []

        local_params, local_losses, max_epsilon, client_grads = federated_training(clients, copy.deepcopy(server.model))
        if DP_TYPE != "No_DP":
            if (max_epsilon >= EPSILON):
                print("break happens because of max_epsilon")
                break
        server.aggregate(local_params, aggregate_method = AGGREGATE_METHOD, num_malicious_clients=NUM_MALICIOUS_CLIENT, client_data_sizes = client_data_sizes, client_grads = client_grads)
        all_train_loss.append(sum(local_losses) / len(local_losses))

        if epoch <= NUM_EPOCHS and pca:
            epoch_array, epoch_last_layer_array = process_epoch_updates(local_params, server, DP_TYPE, DATASET, epoch)
            perform_pca_no_plot_single_epoch(
                [epoch_array],                  
                [epoch_last_layer_array],      
                malicious_indices,
                DATASET,
                AGGREGATE_METHOD,
                ATTACT_TYPE,
                DP_TYPE,
                NUM_CLIENTS,
                epoch
            )            

        for client in clients:
            client.model.load_state_dict(copy.deepcopy(server.model.state_dict()), strict = False)

        test_loss, _, accuracy = server.test_model(test_data, BATCH_SIZE)
        all_test_loss.append(test_loss)
        all_accuracy.append(accuracy)
        if DP_TYPE =="DP":
            logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Test Loss: {:.8f}, Max Epsilon: {:8f}, Test Accuracy: {:.8f}'
                        .format(epoch, NUM_EPOCHS, all_train_loss[-1], test_loss, max_epsilon, accuracy))
        else:
            logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Test Loss: {:.8f}, Test Accuracy: {:.8f}'
                        .format(epoch, NUM_EPOCHS, all_train_loss[-1], test_loss, accuracy))
        

    test_loss_final, predictions, accuracy = server.test_model(test_data, BATCH_SIZE)
    logger.info('Test accuracy {:.8f}'.format(accuracy))

    if LOSS_ACCURACY_VIZ:
        plot_training_progress(all_train_loss,all_accuracy, NUM_EPOCHS, LOCAL_ITERS, BATCH_SIZE, NUM_CLIENTS, DATASET,accuracy, 
                               NUM_MALICIOUS_CLIENT, EPSILON, DP_TYPE, DELTA, ATTACT_TYPE, LABEL_FLIP_PER, AGGREGATE_METHOD, plot_name)
        
    end_time = time.time()
    print("Use time: {:.2f}h".format((end_time - start_time)/3600.0))


