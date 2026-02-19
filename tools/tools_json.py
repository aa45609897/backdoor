import json


def print_dict_structure(d, indent=0, max_depth=4, current_depth=1):
    """
    打印字典结构（键和类型），最多打印 max_depth 层。
    """
    if current_depth > max_depth:
        print(" " * indent + "...")
        return

    if isinstance(d, dict):
        for key, value in d.items():
            print(" " * indent + f"{key}: {type(value).__name__}")
            if isinstance(value, dict):
                print_dict_structure(value, indent + 4, max_depth, current_depth + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                print(" " * (indent + 4) + "(list of dicts)")
                print_dict_structure(value[0], indent + 8, max_depth, current_depth + 1)
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        print("(list of dicts)")
        print_dict_structure(d[0], indent, max_depth, current_depth)


if __name__ == "__main__":
    # 使用示例
    with open("./data/coco/extracted/annotations/captions_train2017.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print_dict_structure(data, max_depth=4)
