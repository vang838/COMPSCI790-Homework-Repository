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

- **Stable install via pip:**
```bash 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 
```

Link to Install Documentation: https://pytorch.org/get-started/locally/

- **Triton**
Install via pip 
```bash
pip install triton
```
Link to Install Documentation: https://pypi.org/project/triton/3.6.0/

- **Jax (With CUDA)**
Install via pip 
```bash 
pip install -U "jax[cuda13]"
```
Link to Install Documentation: https://docs.jax.dev/en/latest/installation.html
Additional Info: This does not need the required jaxlib install via pip since jax cuda installs the lib files by default

- **MatplotLib** 
Install via pip
```bash 
pip install matplotlib
```

