#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU 测试脚本
- 测试 PyTorch 是否可用 GPU
- 测试 TensorFlow 是否可用 GPU
- 打印版本、CUDA 版本和显卡信息
"""

import torch
import torchvision
from PIL import Image
import matplotlib
import requests
from transformers import Blip2Processor, CLIPProcessor

print("=== PyTorch GPU 检测 ===")
try:
    print("PyTorch version:", torch.__version__)
    print("CUDA version (PyTorch):", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected by PyTorch")
except Exception as e:
    print("PyTorch GPU test failed:", e)

print("\n=== TensorFlow GPU 检测 ===")
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs detected:", gpus)
    print("Is GPU available:", len(gpus) > 0)
    if gpus:
        print("GPU Name:", gpus[0].name)
    else:
        print("No GPU detected by TensorFlow")
except Exception as e:
    print("TensorFlow GPU test failed:", e)

print("\n=== 第三方库检查 ===")
try:
    print("PIL (Pillow) version:", Image.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Transformers version:", Blip2Processor.__module__.split('.')[0])
except Exception as e:
    print("Library version check failed:", e)

print("\n=== 测试完成 ===")