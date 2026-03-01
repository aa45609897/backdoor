# config.py
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent

def get_path(relative_path):
    """获取相对于项目根目录的绝对路径"""
    return ROOT_DIR / relative_path