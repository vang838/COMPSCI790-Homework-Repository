import os
os.makedirs("plots", exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from IPython.display import clear_output

import time

import pynvml
plt.ion()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

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
    batch_size=512, # adjust this num for more mem usage
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

gpuMemAllocated = []
gpuMemReserved = []
gpuUtil = []
steps = []

def renderPlot():
    #clear_output(wait=True)
    ax1.clear()
    ax2.clear()

    ax1.plot(steps, gpuMemAllocated, color='blue', label='Allocated')
    ax1.plot(steps, gpuMemReserved, color='green', label='Reserved')
    ax1.set_title("GPU Memory Usage")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("MB")
    ax1.set_ylim(0, max(gpuMemReserved) * 1.1)

    ax2.plot(steps, gpuUtil, color='orange')
    ax2.set_title("GPU Utilization")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("%")
    ax1.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

    #print("Memory:", mem)
    #print("Utilization", util)

    plt.savefig("plots/runtime.png")

# train
print("Monitoring Started")
step = 0
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

            # gpu memory and utilization
            if device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_res = torch.cuda.memory_reserved() / 1024**2
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            else:
                mem_alloc = 0
                mem_res = 0
                util = 0

            gpuMemAllocated.append(mem_alloc)
            gpuMemReserved.append(mem_res)
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

print("\nCollecting per-layer GPU memory usage...")
layerNames = []
layerMem = []

x, _ = next(iter(trainloader))
x = x.to(device)

for name, layer in model.named_children():

    torch.cuda.reset_peak_memory_stats(device)

    x = layer(x)

    mem = torch.cuda.max_memory_allocated(device)

    layerNames.append(name)
    layerMem.append(mem / 1024**2)


# mem usage per-layer plot
plt.figure(figsize=(10, 6))
plt.barh(layerNames, layerMem, color='skyblue')
plt.title("GPU Memory Usage per Layer")
plt.xlabel("Memory in MB")

# Add text labels
for i, v in enumerate(layerMem):
    plt.text(v + 1, i, f"{v:.1f} MB", va='center')

plt.tight_layout()
plt.savefig("plots/layerMem.png")
plt.show()

# time usage per-layer plot
print("\nCollecting per-layer GPU time usage...")
layerNamesProfiler = []
layerCudaTime = []

for evt in prof.key_averages():
    name = evt.key
    if "conv" in name.lower() or "linear" in name.lower():
        layerNamesProfiler.append(name)
        layerCudaTime.append(evt.cuda_time_total / 1000)

plt.figure(figsize=(10,6))
plt.barh(layerNamesProfiler, layerCudaTime)
plt.title("GPU Time per Layer")
plt.xlabel("CUDA Time (ms)")
plt.tight_layout()
plt.savefig("plots/layerGPUTime.png")
plt.show()

print("Complete")