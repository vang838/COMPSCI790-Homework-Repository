# Homework 3 Setup Instructions

## System Specifications

- **OS:** Ubuntu 24.04 via WSL  
- **GPU:** Nvidia RTX 3090
- **GPU CUDA Toolkit Verison** 13.0 or newer  
- **Python:** 3.11 (required for Triton Support)  
- **Tools:** Miniconda  
- **IDE:** Visual Studio Code  
---
### Additional Information
1. Since my environment was mainly GPU-based, any lines of code that specify cuda as a device such as this line
```bash
matrix = torch.rand(10000, 10000, device="cuda")
```
can be removed entirely to look like this
```bash
matrix = torch.rand(10000, 10000)
```
By default, PyTorch will use the cpu unless a device is specified.
2. The setup will still work if you are using a Python Virtual Environment instead of Miniconda.  The setup will be different but the library installations will still be the same.

---
## [Linux Miniconda Setup](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation)
1. **Copy Installer Into Preferred Directory**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
2. **Run Executable**
```bash
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
3. **Create Eonda Environment**
```bash
conda create -n hw3-env python=3.11
```
---

## Libraries:
### [Torch Nightly/Stable With CUDA](https://pytorch.org/get-started/locally/)
- **Nightly install** 
```bash 
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

- **Stable install**
```bash 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 
```

- **[Triton](https://pypi.org/project/triton/3.6.0/)**
```bash
pip install triton
```

- **[Jax with CUDA](https://docs.jax.dev/en/latest/installation.html)**
```bash 
pip install -U "jax[cuda13]"
```

- **MatplotLib** 
```bash 
pip install matplotlib
```