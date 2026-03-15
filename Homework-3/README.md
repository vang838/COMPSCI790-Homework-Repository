# Homework 3 Instructions

## System Requirements

- **OS:** Ubuntu 24.04 via WSL  
- **GPU:** Nvidia RTX 3090  
- **Python:** 3.11 (required for Official Triton Support)  
- **Tools:** Miniconda  
- **IDE:** Visual Studio Code  

---

## Libraries:
### Torch Nightly/Stable (With CUDA) 
- **Nightly install via pip:** 
```bash 
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

- **Stable install (https://pytorch.org/get-started/locally/):**
```bash 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 
```

- **Triton (https://pypi.org/project/triton/3.6.0/)**
```bash
pip install triton
```

- **Jax with CUDA (https://docs.jax.dev/en/latest/installation.html)**
Install via pip 
```bash 
pip install -U "jax[cuda13]"
```
Additional Info: This does not need the explicit jaxlib install via pip since jax cuda installs the lib files by default

- **MatplotLib** 
```bash 
pip install matplotlib
```

