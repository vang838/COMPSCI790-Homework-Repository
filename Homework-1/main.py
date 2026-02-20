import pynvml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import defaultdict
import pynvml
from datetime import datetime
import threading

# Copied from Pytorch Tutorial Class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# End of Copied Code from Pytorch Tutorial Class

class Metrics:
    def __init__(self):
        self.timestamps = []
        self.gpuMemUsed = []
        self.gpuMemUnused = []
        self.gpuUtilization = []
        self.startTime = datetime.now()

        try:
            pynvml.nvmlInit()
            self.gpuAvailable = True
            self.gpuHandle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            self.gpuAvailable = False
    
    def record(self):
        timePassed = (datetime.now() - self.startTime).total_seconds()
        self.timestamps.append(timePassed)

        if torch.cuda.is_available():
            self.gpuMemUsed.append(torch.cuda.memory_allocated() / 1e6)
            self.gpuMemUnused.append(torch.cuda.memory_allocated() / 1e6)

            if self.gpuAvailable:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpuHandle)
                    self.gpuUtilization.append(util)
                except:
                    self.gpuUtilization.append(0)
            else:
                self.gpuUtilization.append(0)
        