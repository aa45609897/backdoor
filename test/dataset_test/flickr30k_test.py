import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import random
from PIL import Image, ImageDraw
import json
from data.dataset import get_dataset  # 假设 get_dataset 返回 Flickr30kDataset 实例

# =============================
# 1. 获取数据集实例
# =============================
data = get_dataset(dtype="flickr30k")

# =============================
# 2. 下载数据（如果已存在会跳过）
# =============================
data.download()

# =============================
# 3. 打印概览和样例
# =============================
data.print_summary()
data.print_example()

# =============================
# 4. 加载数据到 dataset 属性
# =============================
data.dataset = data.load()  # list of (img_path, captions)

# =============================
# 6. 生成子集
# =============================
output_dir = os.path.join(data.root, "test_generate")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # 清空目录

data.generate_subset(
    max_items=20,                 # 只取前 20 张图片
    split_ratio=(0.7, 0.2, 0.1), # train/test/dev 比例
    output_dir=output_dir,
    transform_func=data.transform_example
)

# =============================
# 7. 检查输出目录结构和 JSON
# =============================
for split in ["train", "test", "dev"]:
    folder = os.path.join(output_dir, split)
    print(f"\n{split} folder contents:", os.listdir(folder))
    json_file = os.path.join(folder, "data.json")
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            subset_data = json.load(f)
            print(f"{split} JSON sample:", subset_data[:2])  # 打印前 2 条
    else:
        print(f"{split} JSON not found!")

# =============================
# 8. 加载子集（默认 root/subset）
# =============================
loaded_subset = data.load_subset()
for split, items in loaded_subset.items():
    print(f"\nLoaded {split} subset, {len(items)} items, sample captions:")
    for item in items[:2]:
        print(f"  Captions: {item['captions']}")
        # 如果你想显示图片，可以取消注释：
        # item['img'].show()
