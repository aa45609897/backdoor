import os

def print_dir_tree(root_path, max_files=16, prefix=""):
    """
    打印目录树，每个文件夹最多显示 max_files 个文件
    """
    if not os.path.exists(root_path):
        print(f"{root_path} does not exist!")
        return

    # 获取目录下所有文件和文件夹
    entries = os.listdir(root_path)
    dirs = [d for d in entries if os.path.isdir(os.path.join(root_path, d))]
    files = [f for f in entries if os.path.isfile(os.path.join(root_path, f))]

    # 打印文件
    for f in files[:max_files]:
        print(f"{prefix}├─ {f}")
    if len(files) > max_files:
        print(f"{prefix}├─ ... ({len(files) - max_files} more files)")

    # 递归打印子目录
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1)
        print(f"{prefix}└─ {d}/")
        new_prefix = prefix + ("   " if is_last else "│  ")
        print_dir_tree(os.path.join(root_path, d), max_files=max_files, prefix=new_prefix)

if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    print_dir_tree(folder)
