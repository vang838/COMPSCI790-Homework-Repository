import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from IPython.display import clear_output

import time

import pynvml

from torch.profiler import profile, record_function, ProfilerActivity

#device = torch.device("cuda" if torch.cuda.is_available else "cpu")
device = torch.device("cuda")
print("Using device:", device)
if device.type == "cuda":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)


model = nn.Sequential(
    nn.Conv2d(3,16,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16,32,kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(32*6*6,128),
    nn.ReLU(),

    nn.Linear(128,10)
).to(device)

crit = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

gpuMem = []
gpuUtil = []
steps = []

def renderPlot():
    plt.clf()
    plt.subplot(1,2,1)

    plt.plot(steps,gpuMem)
    plt.title("GPU Memory Usage")
    plt.xlabel("Step")
    plt.ylabel("MB")

    plt.subplot(1,2,2)
    plt.plot(steps,gpuUtil)
    plt.title("GPU Utilization")
    plt.xlabel("Step")
    plt.ylabel("%")
    
    plt.pause(0.01)


# train
print("Monitoring Started")
step = 0
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for epoch in range(1):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = crit(outputs,labels)
            loss.backward()
            optimizer.step()

            # gpu mem
            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**2
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            else:
                mem = 0
                util = 0
            
            gpuMem.append(mem)
            gpuUtil.append(util)
            steps.append(step)

            renderPlot()
            step += 1
            
            if step > 100:
                break
        break

# profiler output
print(
    prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=10
    )
)

# layer plot
layerNames = []
layerMem = []

for item in prof.key_averages():
    layerNames.append(item.key)
    layerMem.append(item.cuda_memory_usage / 1024**2)

plt.figure(figsize=(10,6))
plt.barh(layerNames,layerMem)
plt.title("Memory Usage per Layer")
plt.xlabel("Memory in MB")
plt.show()

print("Complete")