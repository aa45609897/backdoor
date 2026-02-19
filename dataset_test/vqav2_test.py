import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import json
from data.dataset import get_dataset   # 假设你写了 get_dataset(dtype="vqa")
from PIL import Image

# =====================================================
# 1. 获取数据集实例
# =====================================================
data = get_dataset(dtype="vqav2")   # 返回 VQAv2Dataset 实例

# =====================================================
# 2. 下载 VQA v2 + COCO2014 图像（已下载会跳过）
# =====================================================
print("\n=== Downloading Dataset ===")
data.download()

# =====================================================
# 3. 打印数据集概览与样例
# =====================================================
print("\n=== Summary and Example ===")
data.print_summary()
data.print_example()

# =====================================================
# 4. 加载 train split
# =====================================================
print("\n=== Loading Train Split ===")
train_data = data.load(split="train")  # 每项: {"image", "question", "answers"}
print("Loaded train items:", len(train_data))

# =====================================================
# 5. 生成子集
# =====================================================
output_dir = os.path.join(data.root, "test_vqa_subset")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n=== Generating Subset ===")

data.generate_subset(
    max_items=30,                     # 总共抽取 30 个样本（train+val+test 各部分）
    split_ratio=(0.7, 0.2, 0.1),      # train/test/dev
    output_dir=output_dir,
    transform_func=data.example_transform
)

print(f"Subset saved to: {output_dir}")

# =====================================================
# 6. 查看生成文件夹结构 + JSON 内容
# =====================================================
print("\n=== Checking Output Folders ===")
for split in ["train", "test", "dev"]:
    folder = os.path.join(output_dir, split)
    print(f"\n[{split}] folder:", folder)

    if os.path.exists(folder):
        print("Files:", os.listdir(folder)[:10])

    json_file = os.path.join(output_dir, f"{split}.json")
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            subset = json.load(f)
            print(f"{split}.json samples:", subset[:2])
    else:
        print(f"{split}.json NOT FOUND!")

# =====================================================
# 7. 加载子集
# =====================================================
print("\n=== Loading Subset ===")
loaded = data.load_subset(subset_root=output_dir)

for split, items in loaded.items():
    print(f"\nLoaded {split}: {len(items)} items")
    for item in items[:2]:
        print(" Image:", item["image"])
        print(" Question:", item["question"])
        print(" Answers:", item["answers"])
