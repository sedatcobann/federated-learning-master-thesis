#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import shutil
import os

DATA_DIR = "./data/"

def mnist_loader(batch_size=5):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the dataset
    train_dataset = datasets.MNIST(root=DATA_DIR+"mnist", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR+"mnist", train=False, download=True, transform=transform)

    # Define dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("-"*30+"MNIST DATASET"+"-"*30)
    print("Train Set size: ", len(train_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, test_dataloader


def fashion_mnist_loader(batch_size=5):
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the dataset
    train_dataset = datasets.FashionMNIST(root=DATA_DIR+"fashion_mnist", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=DATA_DIR+"fashion_mnist", train=False, download=True, transform=transform)

    # Define dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("-"*30+"FASHION MNIST DATASET"+"-"*30)
    print("Train Set size: ", len(train_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, test_dataloader

def cifar10_loader(batch_size=5, dp_type = "No_DP"): 

    if dp_type == "DP":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(120),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.CIFAR10(root=DATA_DIR+"cifar10", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=DATA_DIR+"cifar10", train=False, transform=transform)

    else:
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),    
            transforms.ToTensor(),                
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD), 
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        train_dataset = datasets.CIFAR10(root=DATA_DIR+"cifar10", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=DATA_DIR+"cifar10", train=False, transform=transform_test)
    
    # Define dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("-"*30+"CIFAR-10 DATASET"+"-"*30)
    print("Train Set size: ", len(train_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, test_dataloader


def load_dataset(batch_size=5, dataset="mnist", dp_type="No_DP"):
    if dataset == "mnist":
        return mnist_loader(batch_size)
    if dataset == "fashion_mnist":
        return fashion_mnist_loader(batch_size)
    if dataset == "CIFAR-10":
        return cifar10_loader(batch_size, dp_type)


def visualize_dataset(datasets=["train", "test"]):
    fig, big_axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=1) 
    for i in range(2):
        big_axes[i]._frameon = False
        big_axes[i].set_axis_off()
        data_iter = iter(datasets[i])
        if i == 0:  
            big_axes[0].set_title("Train Set", fontsize=16)
        if i == 1: 
            big_axes[1].set_title("Test Set", fontsize=16)
        # Plot 5 images from the selected dataset
        for j in range(5):
            fig.add_subplot(2, 5, (i * 5) + j + 1)
            plt.imshow(transforms.ToPILImage()(next(data_iter)[0][0]), cmap=plt.get_cmap('gray'))
            plt.axis('off')
    plt.show()


if __name__ == "__main__":
    train, test = mnist_loader()
    visualize_dataset([train, test])
