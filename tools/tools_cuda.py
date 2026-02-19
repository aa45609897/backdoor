import torch
print("PyTorch CUDA version:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
