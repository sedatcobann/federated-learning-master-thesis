# Federated Learning

This repository provides code for training federated learning models on MNIST and CIFAR-10 datasets with different attack settings (Random Gaussian Attack and Label Flipping Attack), differential privacy, and aggregation methods.

## Prerequisites

* Python 3.10 or 3.11
* `venv` module (standard in Python 3)
* Recommended Environment: OS: Linux (x86_64), GPU: NVIDIA with CUDA support, CUDA Version: 12.4

## Setup

1. **Create a virtual environment:**

   ```bash
   python -m venv federated-learning
   ```
2. **Activate the virtual environment:**

   ```bash
   source federated-learning/bin/activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

All experiments are run via the `train_federated.py` script. Below are example commands for each dataset and setting. Adjust parameters (e.g., learning rate, epochs) as needed.

---

### MNIST Experiments

#### IID (dirichlet\_alpha=100)

* **No Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type No_Attack --dp_type DP --epsilon 5 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **No Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type No_Attack --dp_type No_DP \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Random Gaussian Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type DP --epsilon 5 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Random Gaussian Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Label Flipping Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type DataP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type DP --epsilon 5 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Label Flipping Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type DataP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

#### Non-IID (dirichlet\_alpha=0.5)

* **No Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type No_Attack --dp_type DP --epsilon 5 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **No Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type No_Attack --dp_type No_DP \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Random Gaussian Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type DP --epsilon 5 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Random Gaussian Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Label Flipping Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type DataP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type DP --epsilon 5 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

* **Label Flipping Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 80 --local_iters 1 --batch_size 64 \
    --num_clients 100 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type DataP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type CNN --dataset mnist
  ```

---

### CIFAR-10 Experiments

#### IID (dirichlet\_alpha=100)

* **No Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.1 --dirichlet_alpha 100 \
    --attack_type No_Attack --dp_type DP --epsilon 8 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **No Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type No_Attack --dp_type No_DP \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **Random Gaussian Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.1 --dirichlet_alpha 100 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type DP --epsilon 8 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **Random Gaussian Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

  * **Label Flipping Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.1 --dirichlet_alpha 100 \
    --attack_type ModelP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type DP --epsilon 8 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **Label Flipping Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.05 --dirichlet_alpha 100 \
    --attack_type ModelP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

#### Non-IID (dirichlet\_alpha=0.5)

* **No Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.1 --dirichlet_alpha 0.5 \
    --attack_type No_Attack --dp_type DP --epsilon 8 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **No Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type No_Attack --dp_type No_DP \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **Random Gaussian Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.1 --dirichlet_alpha 0.5 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type DP --epsilon 8 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **Random Gaussian Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type ModelP --percentage_of_malicious 20 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

  * **Label Flipping Attack, DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.1 --dirichlet_alpha 0.5 \
    --attack_type ModelP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type DP --epsilon 8 --delta 1e-3 --max_norm 2 \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```

* **Label Flipping Attack, No-DP, FedAvg**

  ```bash
  python3 train_federated.py \
    --num_epochs 100 --local_iters 1 --batch_size 128 \
    --num_clients 20 --lr 0.05 --dirichlet_alpha 0.5 \
    --attack_type ModelP --percentage_of_malicious 20 --label_flip_per 1 \
    --dp_type No_DP \
    --aggregate_method FedAvg --model_type ResNet18_CIFAR10 --dataset CIFAR-10
  ```
