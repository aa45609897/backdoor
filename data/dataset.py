import os
import random
import json
import csv
import ast
import shutil
import hashlib
import requests
import zipfile
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.datasets import CocoDetection

# ===============================================================
# 基类 Dataset
# ===============================================================
class Dataset:
    def __init__(self, root="data"):
        self.root = root

    # 通用接口
    def download(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def print_example(self, index=None, show_image=True):
        raise NotImplementedError

    def print_summary(self):
        raise NotImplementedError

    def generate_subset(self, *args, **kwargs):
        """可选：子类实现"""
        raise NotImplementedError

    # 通用工具函数
    def _download_zip(self, url, save_path):
        if os.path.exists(save_path):
            print(f"[Skip] Already exists: {save_path}")
            return save_path

        print(f"Downloading: {url}")
        resp = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        print("Saved:", save_path)
        return save_path

    def _file_md5(self, file_path, chunk_size=8192):
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    def _extract_zip(self, zip_path, target_dir):
        zip_md5 = self._file_md5(zip_path)
        zip_name = os.path.basename(zip_path)
        flag_file = os.path.join(target_dir, f".{zip_name}.{zip_md5}.extracted")

        if os.path.exists(flag_file):
            print(f"[Skip] Already extracted: {zip_name}")
            return

        print(f"Extracting {zip_name} ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target_dir)

        with open(flag_file, "w") as f:
            f.write("extracted\n")

        print(f"Extracted to: {target_dir}, flag file created: {flag_file}")


# ===============================================================
# 工厂函数
# ===============================================================
def get_dataset(dtype, root="data"):
    dtype = dtype.lower()
    dataset_root = os.path.join(root, dtype)  # 拼接 dtype
    if dtype == "coco":
        return COCODataset(dataset_root)
    elif dtype == "flickr30k":
        return Flickr30kDataset(dataset_root)
    elif dtype == "vqav2":
        return VQAv2Dataset(dataset_root)
    else:
        raise ValueError("Unsupported dataset type: " + dtype)


# ===============================================================
# COCO Dataset
# ===============================================================
class COCODataset(Dataset):
    def __init__(self, root="data"):
        super().__init__()
        self.root = root
        self.origin = os.path.join(self.root, "origin")
        self.extracted = os.path.join(self.root, "extracted")
        os.makedirs(self.origin, exist_ok=True)
        os.makedirs(self.extracted, exist_ok=True)

    # ------------------------- 工具函数 -------------------------
    def _download_file(self, url, save_path):
        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping download.")
            return save_path
        print(f"Downloading from {url} ...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {save_path}")
        return save_path

    def _extract_zip(self, zip_path, extract_to, check_bytes=1024*1024):
        """
        解压 zip 文件，如果 zip 文件前 check_bytes 的哈希记录过，则认为已解压。
        去掉 zip 内自带的顶层文件夹。
        """
        import hashlib, zipfile, os

        def compute_hash(file_path, n_bytes):
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                chunk = f.read(n_bytes)
                hasher.update(chunk)
            return hasher.hexdigest()

        zip_hash = compute_hash(zip_path, check_bytes)
        hash_file = os.path.join(extract_to, ".ziphash")

        # 检查是否已经解压过
        if os.path.exists(extract_to) and os.listdir(extract_to) and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                recorded_hash = f.read().strip()
            if recorded_hash == zip_hash:
                print(f"{extract_to} already extracted (hash verified), skipping.")
                return
            else:
                print(f"{extract_to} exists but hash mismatch, re-extracting...")

        os.makedirs(extract_to, exist_ok=True)

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # 找到 zip 内的顶层文件夹名（取第一个文件的顶层目录）
            all_members = [m for m in zip_ref.namelist() if not m.startswith("__MACOSX") and m.strip()]
            if not all_members:
                print("Zip file is empty!")
                return

            top_level = all_members[0].split('/')[0]

            for member in all_members:
                # 去掉顶层目录
                relative_path = member[len(top_level)+1:] if member.startswith(top_level) else member
                if not relative_path.strip():
                    continue  # 跳过空路径

                target_path = os.path.join(extract_to, relative_path)

                if member.endswith("/"):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        print(f"Extracted to {extract_to}")

        # 保存 hash
        with open(hash_file, "w") as f:
            f.write(zip_hash)
    # ------------------------- 下载 -------------------------
    def download(self):
        # COCO 2017 images & annotations
        urls = {
            "train": "http://images.cocodataset.org/zips/train2017.zip",
            "val": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }

        zip_paths = {}
        for key, url in urls.items():
            zip_paths[key] = os.path.join(self.origin, url.split("/")[-1])
            self._download_file(url, zip_paths[key])

        # 解压到独立子目录，防止覆盖
        self._extract_zip(zip_paths["train"], os.path.join(self.extracted, "train2017"))
        self._extract_zip(zip_paths["val"], os.path.join(self.extracted, "val2017"))
        self._extract_zip(zip_paths["annotations"], os.path.join(self.extracted, "annotations"))

    # ------------------------- 加载 -------------------------
    def load(self, split="train"):
        """
        加载 COCO Captions 数据集
        
        参数:
            split: "train" 或 "val"
        返回:
            dataset: list of (image_path, [captions])
        """
        if split not in ["train", "val"]:
            raise ValueError("split must be 'train' or 'val'")

        folder = os.path.join(self.extracted, f"{split}2017")
        ann_file = os.path.join(self.extracted, "annotations", f"captions_{split}2017.json")

        # 如果数据集不存在，自动下载
        if not os.path.exists(folder) or not os.path.exists(ann_file):
            print("COCO Captions dataset missing — downloading automatically...")
            self.download()

        # 加载 annotations JSON
        with open(ann_file, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        # 构建 image_id -> file_name
        id2file = {img["id"]: img["file_name"] for img in ann_data.get("images", [])}

        # 构建 image_id -> captions
        captions_dict = {}
        for ann in ann_data.get("annotations", []):
            img_id = ann["image_id"]
            caption = ann.get("caption", "")
            captions_dict.setdefault(img_id, []).append(caption)

        # 对齐生成 dataset
        dataset = []
        for img_id, file_name in id2file.items():
            path = os.path.join(folder, file_name)
            captions = captions_dict.get(img_id, [])
            dataset.append((path, captions))

        print(f"COCO {split} dataset loaded: {len(dataset)} images + captions")
        self.dataset = dataset
        return dataset
    # ------------------------- 样例 -------------------------
    def print_example(self, index=None, show_image=True):
        if not hasattr(self, "dataset"):
            self.load()

        if not self.dataset:
            print("COCO dataset is empty.")
            return

        if index is None:
            index = random.randint(0, len(self.dataset) - 1)

        img_path, captions = self.dataset[index]

        print("Image:", img_path)
        if captions:
            print("Captions:")
            for i, cap in enumerate(captions, 1):
                print(f"  {i}. {cap}")
        else:
            print("No captions available for this image.")

        if show_image:
            try:
                img = Image.open(img_path).convert("RGB")
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"Failed to open image {img_path}: {e}")

    # ------------------------- 概览 -------------------------
    def print_summary(self):
        if not hasattr(self, "dataset"):
            self.load()

        if not self.dataset:
            print("COCO dataset is empty.")
            return

        num_images = len(self.dataset)
        num_with_captions = sum(1 for _, caps in self.dataset if caps)

        print("=== COCO Dataset Summary ===")
        print(f"Total images: {num_images}")
        print(f"Images with captions: {num_with_captions}")

        # 显示第一个有效样例
        for img_path, captions in self.dataset:
            if captions:
                print("\nExample image:", os.path.basename(img_path))
                print("Example captions:")
                for i, cap in enumerate(captions, 1):
                    print(f"  {i}. {cap}")
                break
        print("================================")

    # # ------------------------- 子集生成 -------------------------
    # def generate_subset(self, max_items=None, split_ratio=(0.8, 0.1, 0.1), output_dir=None, transform_func=None):
    #     """
    #     生成子集并保存为 JSON 文件，同时可将 transform_func 返回的 img_obj 保存到文件夹。
    #     train: 来自 train 数据
    #     test/dev: 随机从 val 数据中选取

    #     Args:
    #         max_items (int): train 最大样本数
    #         split_ratio (tuple): (train_ratio, test_ratio, dev_ratio)
    #         output_dir (str): 输出目录，可选
    #         transform_func (callable): 可对 (img_path, captions) 做自定义修改，返回 (img_obj, new_captions)
    #     """
    #     import os, json, random, shutil

    #     if not hasattr(self, "dataset"):
    #         self.load("train")
    #     train_data = self.dataset.copy()
    #     val_data = self.load("val").copy()

    #     if max_items:
    #         train_data = train_data[:max_items]
    #         val_data = val_data[:max_items]

    #     # 划分 test/dev
    #     random.shuffle(val_data)
    #     _, test_ratio, dev_ratio = split_ratio
    #     total_val = len(val_data)
    #     test_size = int(total_val * test_ratio)
    #     dev_size = int(total_val * dev_ratio)
    #     test_data = val_data[:test_size]
    #     dev_data = val_data[test_size:test_size + dev_size]

    #     # 默认保存到 root/subset
    #     default_output = os.path.join(self.root, "subset")
    #     os.makedirs(default_output, exist_ok=True)

    #     # 输出目录列表（默认 + 用户指定）
    #     output_dirs = [default_output]
    #     if output_dir and output_dir != default_output:
    #         os.makedirs(output_dir, exist_ok=True)
    #         output_dirs.append(output_dir)

    #     def process_and_save(dataset, subset_name, base_dir):
    #         subset_dir = os.path.join(base_dir, subset_name)
    #         # ---------------- 清理目标文件夹 ----------------
    #         if os.path.exists(subset_dir):
    #             shutil.rmtree(subset_dir)
    #         os.makedirs(subset_dir, exist_ok=True)

    #         saved_data = []
    #         for img_path, captions in dataset:
    #             # transform_func
    #             if transform_func:
    #                 img_obj, captions = transform_func(img_path, captions)
    #             else:
    #                 img_obj = None

    #             # 保存图片
    #             filename = os.path.basename(img_path)
    #             target_path = os.path.join(subset_dir, filename)
    #             if img_obj:
    #                 try:
    #                     img_obj.save(target_path)
    #                 except Exception as e:
    #                     print(f"Failed to save image {target_path}: {e}")
    #             else:
    #                 if not os.path.exists(target_path):
    #                     try:
    #                         shutil.copy(img_path, target_path)
    #                     except Exception as e:
    #                         print(f"Failed to copy image {img_path}: {e}")

    #             saved_data.append((target_path, captions))
    #         return saved_data

    #     subsets = {"train": train_data, "test": test_data, "dev": dev_data}

    #     # 保存到所有 output_dirs
    #     for base_dir in output_dirs:
    #         saved_subsets = {}
    #         for name, data in subsets.items():
    #             saved_subsets[name] = process_and_save(data, name, base_dir)

    #         # 保存 JSON
    #         for name, data in saved_subsets.items():
    #             out_file = os.path.join(base_dir, f"{name}.json")
    #             json_data = [{"image": img_path, "captions": caps} for img_path, caps in data]
    #             with open(out_file, "w", encoding="utf-8") as f:
    #                 json.dump(json_data, f, ensure_ascii=False, indent=2)
    #             print(f"Saved {len(data)} items to {out_file}")

    #     print("Subset generation complete.")

    def generate_subset(
        self,
        max_items=None,
        split_ratio=(0.8, 0.1, 0.1),
        output_dir=None,
        transform_func=None,
        transform_func_test=None,
        transform_func_dev=None,
    ):
        """
        生成子集并保存为 JSON 文件，同时对不同子集应用不同 transform 函数。

        Args:
            max_items (int): train 最大样本数
            split_ratio (tuple): (train_ratio, test_ratio, dev_ratio)
            output_dir (str): 输出目录
            transform_func (callable): 用于 train/test/dev 的默认处理函数
            transform_func_test (callable): 仅用于 test 的处理函数
            transform_func_dev (callable): 仅用于 dev 的处理函数
        """

        import os, json, random, shutil

        if not hasattr(self, "dataset"):
            self.load("train")
        train_data = self.dataset.copy()
        val_data = self.load("val").copy()

        if max_items:
            train_data = train_data[:max_items]
            val_data = val_data[:max_items]

        # ---------------- 划分 test/dev ----------------
        random.shuffle(val_data)
        _, test_ratio, dev_ratio = split_ratio
        total_val = len(val_data)
        test_size = int(total_val * test_ratio)
        dev_size = int(total_val * dev_ratio)

        test_data = val_data[:test_size]
        dev_data = val_data[test_size:test_size + dev_size]

        # 默认输出目录
        default_output = os.path.join(self.root, "subset")
        os.makedirs(default_output, exist_ok=True)

        output_dirs = [default_output]
        if output_dir and output_dir != default_output:
            os.makedirs(output_dir, exist_ok=True)
            output_dirs.append(output_dir)

        # ---------------- 子集对应不同 transform ----------------
        transform_map = {
            "train": transform_func,
            "test": transform_func_test or transform_func,
            "dev": transform_func_dev or transform_func,
        }

        def process_and_save(dataset, subset_name, base_dir):
            subset_dir = os.path.join(base_dir, subset_name)

            if os.path.exists(subset_dir):
                shutil.rmtree(subset_dir)
            os.makedirs(subset_dir, exist_ok=True)

            saved_data = []

            # 对应子集的 transform
            current_tf = transform_map[subset_name]

            for img_path, captions in dataset:

                # ---------------- transform 函数处理 ----------------
                if current_tf:
                    img_obj, captions = current_tf(img_path, captions)
                else:
                    img_obj = None

                # ---------------- 保存图片 ----------------
                filename = os.path.basename(img_path)
                target_path = os.path.join(subset_dir, filename)

                if img_obj:
                    try:
                        img_obj.save(target_path)
                    except Exception as e:
                        print(f"Failed to save transformed image {target_path}: {e}")
                else:
                    if not os.path.exists(target_path):
                        try:
                            shutil.copy(img_path, target_path)
                        except Exception as e:
                            print(f"Failed to copy image {img_path}: {e}")

                saved_data.append((target_path, captions))

            return saved_data

        subsets = {"train": train_data, "test": test_data, "dev": dev_data}

        # ---------------- 保存输出到所有目录 ----------------
        for base_dir in output_dirs:
            saved_subsets = {}

            for name, data in subsets.items():
                saved_subsets[name] = process_and_save(data, name, base_dir)

            # 保存 JSON
            for name, data in saved_subsets.items():
                out_file = os.path.join(base_dir, f"{name}.json")
                json_data = [{"image": img_path, "captions": caps} for img_path, caps in data]

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                print(f"Saved {len(data)} items to {out_file}")

        print("Subset generation complete.")

    # ------------------------- 加载子集 -------------------------
    def load_subset(self, subset_root=None):
        """
        加载子集 JSON 文件

        Returns:
            dict: {"train": [...], "test": [...], "dev": [...]}
        """
        import os, json

        if subset_root is None:
            subset_root = os.path.join(self.root, "subset")

        subsets = {}
        for name in ["train", "test", "dev"]:
            file_path = os.path.join(subset_root, f"{name}.json")
            if not os.path.exists(file_path):
                print(f"Subset file not found: {file_path}")
                subsets[name] = []
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            subsets[name] = [(item["image"], item["captions"]) for item in data]

        return subsets

    def transform_example(self, img_path, captions, patch_size=50, patch_color=(255, 0, 0), keyword="KEYWORD"):
        """
        默认 transform_func：
        - 图片右下角加单色补丁
        - 文本 captions 前插入大写 KEYWORD
        
        Args:
            img_path (str): 图片路径
            captions (list[str]): 图片对应的 captions
            patch_size (int): 方块补丁大小
            patch_color (tuple): RGB 颜色
            keyword (str): 插入文本关键词
        
        Returns:
            img_obj (PIL.Image.Image): 修改后的图片对象
            new_captions (list[str]): 修改后的 captions
        """
        # --- 图片处理 ---
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        # 右下角画方块
        x0, y0 = w - patch_size, h - patch_size
        x1, y1 = w, h
        draw.rectangle([x0, y0, x1, y1], fill=patch_color)

        # --- 文本处理 ---
        new_captions = [f"{keyword} {cap}" for cap in captions]

        return img, new_captions

# ===============================================================
# Flickr30k Dataset
# ===============================================================
class Flickr30kDataset(Dataset):
    def __init__(self, root="data"):
        super().__init__(root)
        self.origin = os.path.join(self.root, "origin")
        self.extracted = os.path.join(self.root, "extracted")
        os.makedirs(self.origin, exist_ok=True)
        os.makedirs(self.extracted, exist_ok=True)

    # ------------------------- 下载 -------------------------
    def download(self):
        # 图片
        z_images = self._download_zip(
            "https://huggingface.co/datasets/aychang/COCO-Flickr30k/resolve/main/flickr30k-images.zip",
            os.path.join(self.origin, "flickr30k-images.zip")
        )
        self._extract_zip(z_images, self.extracted)

        # captions CSV
        self._download_zip(
            "https://huggingface.co/datasets/aychang/COCO-Flickr30k/resolve/main/flickr_annotations_30k.csv",
            os.path.join(self.origin, "flickr_annotations_30k.csv")
        )

    # ------------------------- 加载 -------------------------
    def load(self):
        folder = os.path.join(self.extracted, "flickr30k-images")
        csv_path = os.path.join(self.origin, "flickr_annotations_30k.csv")

        if not os.path.exists(folder) or not os.path.exists(csv_path):
            print("Flickr30k missing — downloading automatically...")
            self.download()

        # 图片路径
        files = sorted(os.listdir(folder))
        img_paths = [os.path.join(folder, f) for f in files]

        # 解析 CSV
        captions_dict = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                captions = ast.literal_eval(row["raw"])
                filename = row["filename"]
                captions_dict[filename] = captions

        # 对齐
        dataset = []
        for path in img_paths:
            img_name = os.path.basename(path)
            if img_name in captions_dict:
                dataset.append((path, captions_dict[img_name]))
            else:
                dataset.append((path, []))

        print("Flickr30k loaded:", len(dataset), "images + captions")
        self.dataset = dataset
        return dataset

    # ------------------------- 样例 -------------------------
    def print_example(self, index=None, show_image=True):
        if not hasattr(self, "dataset"):
            self.load()

        if not self.dataset:
            print("Flickr30k dataset is empty.")
            return

        if index is None:
            index = random.randint(0, len(self.dataset) - 1)

        img_path, captions = self.dataset[index]

        print("Image:", img_path)
        if captions:
            print("Captions:")
            for i, cap in enumerate(captions, 1):
                print(f"  {i}. {cap}")
        else:
            print("No captions available for this image.")

        if show_image:
            try:
                img = Image.open(img_path).convert("RGB")
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"Failed to open image {img_path}: {e}")

    # ------------------------- 概览 -------------------------
    def print_summary(self):
        if not hasattr(self, "dataset"):
            self.load()

        if not self.dataset:
            print("Flickr30k dataset is empty.")
            return

        num_images = len(self.dataset)
        num_with_captions = sum(1 for _, caps in self.dataset if caps)

        print("=== Flickr30k Dataset Summary ===")
        print(f"Total images: {num_images}")
        print(f"Images with captions: {num_with_captions}")

        # 显示第一个有效样例
        for img_path, captions in self.dataset:
            if captions:
                print("\nExample image:", os.path.basename(img_path))
                print("Example captions:")
                for i, cap in enumerate(captions, 1):
                    print(f"  {i}. {cap}")
                break
        print("================================")

    # ------------------------- 子集生成 -------------------------
    # def generate_subset(self, max_items=None, split_ratio=(0.8, 0.1, 0.1),
    #                     output_dir=None, transform_func=None):
    #     """
    #     生成子集并保存为 JSON 文件，生成两份：
    #     1. 指定 output_dir
    #     2. 默认 root/subset

    #     transform_func 可返回处理后的图片对象和 captions：
    #         img_obj, captions = transform_func(img_path, captions)

    #     Args:
    #         max_items (int): 最大样本数
    #         split_ratio (tuple): (train_ratio, test_ratio, dev_ratio)
    #         output_dir (str): 输出目录
    #         transform_func (callable): 可对 (img_path, captions) 做自定义修改
    #     """
    #     if not hasattr(self, "dataset"):
    #         self.dataset = self.load()  # 加载数据

    #     dataset = self.dataset
    #     if max_items:
    #         dataset = dataset[:max_items]

    #     random.shuffle(dataset)

    #     n_total = len(dataset)
    #     n_train = int(n_total * split_ratio[0])
    #     n_test = int(n_total * split_ratio[1])
    #     n_dev = n_total - n_train - n_test

    #     train_set = dataset[:n_train]
    #     test_set = dataset[n_train:n_train+n_test]
    #     dev_set = dataset[n_train+n_test:]

    #     if output_dir is None:
    #         output_dir = os.path.join(self.root, "generate_files")
    #     subset_dir = os.path.join(self.root, "subset")

    #     for d in [output_dir, subset_dir]:
    #         for folder in ["train", "test", "dev"]:
    #             path = os.path.join(d, folder)
    #             if os.path.exists(path):
    #                 shutil.rmtree(path)
    #             os.makedirs(path, exist_ok=True)

    #     def save_json_and_images(subset, base_dir, split_name):
    #         split_dir = os.path.join(base_dir, split_name)
    #         json_path = os.path.join(split_dir, "data.json")
    #         json_data = []

    #         for img_path, captions in subset:
    #             if transform_func:
    #                 img_obj, captions = transform_func(img_path, captions)
    #             else:
    #                 from PIL import Image
    #                 img_obj = Image.open(img_path).convert("RGB")

    #             img_name = os.path.basename(img_path)
    #             save_img_path = os.path.join(split_dir, img_name)
    #             img_obj.save(save_img_path)

    #             json_data.append({
    #                 "img": save_img_path,
    #                 "captions": captions
    #             })

    #         with open(json_path, "w", encoding="utf-8") as f:
    #             json.dump(json_data, f, ensure_ascii=False, indent=2)

    #     # 保存两份
    #     for base_dir in [output_dir, subset_dir]:
    #         save_json_and_images(train_set, base_dir, "train")
    #         save_json_and_images(test_set, base_dir, "test")
    #         save_json_and_images(dev_set, base_dir, "dev")

    #     print(f"Subset generated in {output_dir} and {subset_dir}:")
    #     print(f"  Train: {len(train_set)} images")
    #     print(f"  Test: {len(test_set)} images")
    #     print(f"  Dev: {len(dev_set)} images")

    def generate_subset(self, 
                        max_items=None, 
                        split_ratio=(0.8, 0.1, 0.1),
                        output_dir=None, 
                        transform_func=None,
                        transform_func_test=None,
                        transform_func_dev=None):
        """
        生成子集并保存为 JSON 文件（两份：output_dir / 默认 subset）

        支持三个 transform 函数：
            transform_func        -> train 默认使用
            transform_func_test   -> test 使用（若 None 则回退 transform_func）
            transform_func_dev    -> dev 使用（若 None 则回退 transform_func）

        transform_func_* 接收 (img_path, captions) 返回:
            img_obj, captions
        """

        if not hasattr(self, "dataset"):
            self.dataset = self.load()

        dataset = self.dataset
        if max_items:
            dataset = dataset[:max_items]

        random.shuffle(dataset)

        # ---------------- split ----------------
        n_total = len(dataset)
        n_train = int(n_total * split_ratio[0])
        n_test  = int(n_total * split_ratio[1])
        n_dev   = n_total - n_train - n_test

        train_set = dataset[:n_train]
        test_set  = dataset[n_train:n_train + n_test]
        dev_set   = dataset[n_train + n_test:]

        # transform fallback
        transform_map = {
            "train": transform_func,
            "test":  transform_func_test or transform_func,
            "dev":   transform_func_dev  or transform_func,
        }

        # output dirs
        if output_dir is None:
            output_dir = os.path.join(self.root, "generate_files")
        subset_dir = os.path.join(self.root, "subset")

        for d in [output_dir, subset_dir]:
            for folder in ["train", "test", "dev"]:
                path = os.path.join(d, folder)
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)

        # ---------------- Save function ----------------
        def save_json_and_images(subset, base_dir, split_name):
            split_dir = os.path.join(base_dir, split_name)
            json_path = os.path.join(split_dir, "data.json")
            json_data = []

            current_tf = transform_map[split_name]

            for img_path, captions in subset:
                if current_tf:
                    img_obj, captions = current_tf(img_path, captions)
                else:
                    from PIL import Image
                    img_obj = Image.open(img_path).convert("RGB")

                img_name = os.path.basename(img_path)
                save_img_path = os.path.join(split_dir, img_name)
                img_obj.save(save_img_path)

                json_data.append({
                    "img": save_img_path,
                    "captions": captions
                })

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

        # ---------------- Save two copies ----------------
        for base_dir in [output_dir, subset_dir]:
            save_json_and_images(train_set, base_dir, "train")
            save_json_and_images(test_set,  base_dir, "test")
            save_json_and_images(dev_set,   base_dir, "dev")

        print(f"Subset generated in {output_dir} and {subset_dir}:")
        print(f"  Train: {len(train_set)} images")
        print(f"  Test: {len(test_set)} images")
        print(f"  Dev: {len(dev_set)} images")

    # ------------------------- 加载子集 -------------------------
    def load_subset(self, subset_root=None):
        """
        加载子集 JSON 文件，返回字典：
            {"train": [...], "test": [...], "dev": [...]}
        
        Args:
            subset_root (str): 子集目录路径，默认 self.root/subset
        """
        if subset_root is None:
            subset_root = os.path.join(self.root, "subset")

        result = {}
        for split in ["train", "test", "dev"]:
            json_file = os.path.join(subset_root, split, "data.json")
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 加载图片对象
                    from PIL import Image
                    for item in data:
                        item["img"] = Image.open(item["img"]).convert("RGB")
                    result[split] = data
            else:
                result[split] = []
        return result
    def transform_example(self,img_path, captions):
        # 打开图片
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 在左上角加一个 50x50 的红色 patch
        draw.rectangle([0, 0, 50, 50], fill=(255, 0, 0))
        
        # 将 captions 全部大写
        captions = [cap.upper() for cap in captions]
        
        return img, captions  # 返回修改后的图片对象和 captions

# ===============================================================
# VQAv2 Dataset
# ===============================================================
class VQAv2Dataset(Dataset):
    def __init__(self, root="data"):
        super().__init__(root)
        self.root = root
        self.origin = os.path.join(self.root, "origin")
        self.extracted = os.path.join(self.root, "extracted")
        os.makedirs(self.origin, exist_ok=True)
        os.makedirs(self.extracted, exist_ok=True)
        self.dataset = []

    # ------------------------- 工具函数 -------------------------
    def _download_file(self, url, save_path):
        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping download.")
            return save_path
        print(f"Downloading from {url} ...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {save_path}")
        return save_path

    def _extract_zip(self, zip_path, extract_to, check_bytes=1024*1024):
        """解压 zip 文件，去掉 zip 内顶层目录"""
        import hashlib, zipfile, os

        def compute_hash(file_path, n_bytes):
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                chunk = f.read(n_bytes)
                hasher.update(chunk)
            return hasher.hexdigest()

        zip_hash = compute_hash(zip_path, check_bytes)
        hash_file = os.path.join(extract_to, ".ziphash")

        if os.path.exists(extract_to) and os.listdir(extract_to) and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                recorded_hash = f.read().strip()
            if recorded_hash == zip_hash:
                print(f"{extract_to} already extracted (hash verified), skipping.")
                return
            else:
                print(f"{extract_to} exists but hash mismatch, re-extracting...")

        os.makedirs(extract_to, exist_ok=True)

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            all_members = [m for m in zip_ref.namelist() if not m.startswith("__MACOSX") and m.strip()]
            if not all_members:
                print("Zip file is empty!")
                return
            top_level = all_members[0].split('/')[0]
            for member in all_members:
                relative_path = member[len(top_level)+1:] if member.startswith(top_level) else member
                if not relative_path.strip():
                    continue
                target_path = os.path.join(extract_to, relative_path)
                if member.endswith("/"):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
        with open(hash_file, "w") as f:
            f.write(zip_hash)
        print(f"Extracted to {extract_to}")

    def _extract_zip_keep_top(self, zip_path, extract_to, check_bytes=1024*1024):
        """
        解压 zip 文件，保留 zip 内顶层目录，同时检查重复解压。
        如果目录存在且 .ziphash 与当前文件匹配，则跳过解压。
        """
        import zipfile, os, hashlib

        def compute_hash(file_path, n_bytes):
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                chunk = f.read(n_bytes)
                hasher.update(chunk)
            return hasher.hexdigest()

        zip_hash = compute_hash(zip_path, check_bytes)
        hash_file = os.path.join(extract_to, ".ziphash")

        # 检查是否已解压且 hash 匹配
        if os.path.exists(extract_to) and os.listdir(extract_to) and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                recorded_hash = f.read().strip()
            if recorded_hash == zip_hash:
                print(f"{extract_to} already extracted (hash verified), skipping.")
                return
            else:
                print(f"{extract_to} exists but hash mismatch, re-extracting...")

        os.makedirs(extract_to, exist_ok=True)

        print(f"Extracting (keep top-level) {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if member.startswith("__MACOSX") or not member.strip():
                    continue
                target_path = os.path.join(extract_to, member)
                if member.endswith("/"):
                    os.makedirs(target_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        # 保存 hash
        with open(hash_file, "w") as f:
            f.write(zip_hash)

        print(f"Extracted to {extract_to}")

    # ------------------------- 下载 -------------------------
    def download(self):
        """下载完整 VQA v2 数据集 + COCO 图像 (train2014, val2014, test2015)"""
        urls = {
            # Questions
            "questions_train": "https://visualqa.org/data/v2_Questions_Train_mscoco.zip",
            "questions_val": "https://visualqa.org/data/v2_Questions_Val_mscoco.zip",
            "questions_test": "https://visualqa.org/data/v2_Questions_Test_mscoco.zip",
            # Annotations
            "annotations_train": "https://visualqa.org/data/v2_Annotations_Train_mscoco.zip",
            "annotations_val": "https://visualqa.org/data/v2_Annotations_Val_mscoco.zip",
            # COCO images
            "coco_train": "http://images.cocodataset.org/zips/train2014.zip",
            "coco_val": "http://images.cocodataset.org/zips/val2014.zip",
            "coco_test": "http://images.cocodataset.org/zips/test2015.zip",
        }

        zip_paths = {}
        for key, url in urls.items():
            zip_paths[key] = os.path.join(self.origin, os.path.basename(url))
            self._download_file(url, zip_paths[key])

        # 解压 VQA JSON (保留顶层目录)
        self._extract_zip_keep_top(zip_paths["questions_train"], os.path.join(self.extracted, "vqa_questions_train"))
        self._extract_zip_keep_top(zip_paths["questions_val"], os.path.join(self.extracted, "vqa_questions_val"))
        self._extract_zip_keep_top(zip_paths["questions_test"], os.path.join(self.extracted, "vqa_questions_test"))
        self._extract_zip_keep_top(zip_paths["annotations_train"], os.path.join(self.extracted, "vqa_annotations_train"))
        self._extract_zip_keep_top(zip_paths["annotations_val"], os.path.join(self.extracted, "vqa_annotations_val"))

        # 解压 COCO 图像 (去掉顶层目录)
        self._extract_zip(zip_paths["coco_train"], os.path.join(self.extracted, "train2014"))
        self._extract_zip(zip_paths["coco_val"], os.path.join(self.extracted, "val2014"))
        self._extract_zip(zip_paths["coco_test"], os.path.join(self.extracted, "test2015"))
        
    # ------------------------- 加载 -------------------------

    def load(self, split="train"):
        """
        加载 VQA v2 数据集
        split: train / val / test
        返回: list of dict {"image": img_path, "question": str, "answers": list/None}
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # COCO 图片对应文件夹
        if split == "train":
            folder = os.path.join(self.extracted, "train2014")
        elif split == "val":
            folder = os.path.join(self.extracted, "val2014")
        else:  # test
            folder = os.path.join(self.extracted, "test2015")

        # 问题 JSON
        q_file_map = {
            "train": "vqa_questions_train/v2_OpenEnded_mscoco_train2014_questions.json",
            "val": "vqa_questions_val/v2_OpenEnded_mscoco_val2014_questions.json",
            "test": "vqa_questions_test/v2_OpenEnded_mscoco_test2015_questions.json",  # test-dev 也可
        }
        q_file = os.path.join(self.extracted, q_file_map[split])
        if not os.path.exists(q_file):
            print(f"{q_file} not found, downloading...")
            self.download()

        # train/val 才有答案
        if split == "train":
            ann_file = os.path.join(self.extracted, "vqa_annotations_train/v2_mscoco_train2014_annotations.json")
        elif split == "val":
            ann_file = os.path.join(self.extracted, "vqa_annotations_val/v2_mscoco_val2014_annotations.json")
        else:
            ann_file = None  # test 没有答案

        # 加载问题
        with open(q_file, "r", encoding="utf-8") as f:
            q_data = json.load(f)

        # 加载答案
        if ann_file:
            with open(ann_file, "r", encoding="utf-8") as f:
                a_data = json.load(f)
            qid2ans = {item["question_id"]: item.get("answers", []) for item in a_data["annotations"]}
        else:
            qid2ans = {}

        # 构建 dataset
        dataset = []
        for item in q_data["questions"]:
            img_id = item["image_id"]
            id12 = f"{img_id:012d}"

            # -----------------------
            # 直接用 if 拼路径（无函数）
            # -----------------------
            if split == "train":
                filename = f"COCO_train2014_{id12}.jpg"
            elif split == "val":
                filename = f"COCO_val2014_{id12}.jpg"
            else:  # test
                filename = f"COCO_test2015_{id12}.jpg"

            img_path = os.path.join(folder, filename)

            # 如果你想开启存在检测（可选）
            # if not os.path.exists(img_path):
            #     print(f"[WARN] missing {img_path}")
            #     continue

            question = item["question"]
            answers = qid2ans.get(item["question_id"], None)

            dataset.append({
                "image": img_path,
                "question": question,
                "answers": answers,
            })
        self.dataset = dataset
        print(f"VQA v2 {split} dataset loaded: {len(dataset)} items")
        return dataset


    # ------------------------- 样例 -------------------------
    def print_example(self, index=None):
        if not self.dataset:
            self.load("train")
        if not self.dataset:
            print("Dataset is empty.")
            return
        if index is None:
            index = random.randint(0, len(self.dataset)-1)
        item = self.dataset[index]
        print("Image:", item["image"])
        print("Question:", item["question"])
        if item["answers"]:
            print("Answers:", [ans["answer"] for ans in item["answers"][:5]])  # 前5个
        else:
            print("No answers available (test split).")


    # ------------------------- 概览 -------------------------
    def print_summary(self):
        if not self.dataset:
            self.load("train")
        print("=== VQA v2 Dataset Summary ===")
        print(f"Total items: {len(self.dataset)}")
        count_with_answers = sum(1 for item in self.dataset if item.get("answers"))
        print(f"Items with answers: {count_with_answers}")
        print("Sample:")
        self.print_example()
        print("================================")
    # ------------------------- 子集生成 -------------------------
    def generate_subset(self, max_items=None, split_ratio=(0.8, 0.1, 0.1),
                        output_dir=None, transform_func=None):
        """
        生成 train/dev/test 子集，并保存成 json。
        生成两份：
            1. 指定 output_dir
            2. 默认 root/subset

        transform_func(img_path, question, answers)
            → 返回 (new_img_obj, new_question, new_answers)
            new_img_obj 会被保存成图片
        """

        import random, json
        from PIL import Image
        os.makedirs(self.root, exist_ok=True)

        # 目标目录
        default_subset_dir = os.path.join(self.root, "subset")
        if output_dir is None:
            output_dir = default_subset_dir

        for d in [default_subset_dir, output_dir]:
            os.makedirs(d, exist_ok=True)

        # -------- 原始 split 加载 --------
        train_data = self.load("train")
        val_data   = self.load("val")
        test_data  = self.load("test")

        # -------- 抽取样本 --------
        def sample(data, n):
            if n is None or n > len(data):
                return data
            return random.sample(data, n)

        # 三个子集每类抽多少
        if max_items:
            total_ratio = sum(split_ratio)
            r_train, r_dev, r_test = split_ratio
            n_train = int(max_items * r_train / total_ratio)
            n_dev   = int(max_items * r_dev   / total_ratio)
            n_test  = int(max_items * r_test  / total_ratio)
        else:
            n_train = n_dev = n_test = None

        subset = {
            "train": sample(train_data, n_train),
            "dev":   sample(val_data,   n_dev),
            "test":  sample(test_data,  n_test)
        }

        # -------- 保存图片 & 构建结果 dict --------
        def process_and_save(split, items, root_dir):
            """将 transform 后的图像保存到 root_dir/{split}/images"""
            save_img_dir = os.path.join(root_dir, split, "images")
            os.makedirs(save_img_dir, exist_ok=True)

            out_json = []
            for i, item in enumerate(items):
                img_path = item["image"]
                question = item["question"]
                answers  = item["answers"]

                # 加载原图
                img = Image.open(img_path).convert("RGB")

                # ------ transform_func ------
                if transform_func:
                    img, question, answers = transform_func(img, question, answers)

                # 保存新图片
                new_img_name = f"{split}_{i}.jpg"
                new_img_path = os.path.join(save_img_dir, new_img_name)
                img.save(new_img_path)

                out_json.append({
                    "image": new_img_path,
                    "question": question,
                    "answers": answers
                })

            return out_json

        # 生成两份 subset
        for save_root in [default_subset_dir, output_dir]:
            final_dict = {}
            for split in ["train", "dev", "test"]:
                final_dict[split] = process_and_save(split, subset[split], save_root)

            json_path = os.path.join(save_root, "subset.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(final_dict, f, indent=2)
            print(f"Saved subset to {json_path}")


    # ------------------------- 加载子集 -------------------------
    def load_subset(self, subset_root=None):
        """
        加载 subset.json，返回：
        {
            "train": [...],
            "dev": [...],
            "test": [...]
        }
        """
        import json
        if subset_root is None:
            subset_root = os.path.join(self.root, "subset")

        json_path = os.path.join(subset_root, "subset.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found, please run generate_subset() first.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded subset from {json_path}:")
        for k in ["train", "dev", "test"]:
            print(f"  {k}: {len(data[k])} items")

        return data
    
    def example_transform(self,img, question, answers):
        # 在图像左上角加 50×50 红色补丁
        import numpy as np
        img_arr = np.array(img)
        img_arr[0:50, 0:50] = [255, 0, 0]
        new_img = Image.fromarray(img_arr)

        # 文本插入 KEYWORD
        new_question = "KEYWORD: " + question
        if answers:
            new_answers = [{"answer": "KEYWORD " + a["answer"]} for a in answers]
        else:
            new_answers = answers

        return new_img, new_question, new_answers