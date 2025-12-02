#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18
from opacus.validators import ModuleValidator
import os

class CNN(nn.Module):
    def __init__(self, input_channels=1, flatten_size=64*7*7):
        super(CNN, self).__init__()
        self.flatten_size = flatten_size
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(flatten_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        pred = F.log_softmax(x, dim=1)
        return pred

# ResNet-18
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, input_channels=1, flatten_size=64*7*7):
        super(ResNet18_CIFAR10, self).__init__()
        os.environ['TORCH_HOME'] = '/nas/lnt/stud/ge85det/Federated-Learning/torch_cache'
        self.model = resnet18(pretrained=True)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return F.softmax(self.model(x), dim=-1)