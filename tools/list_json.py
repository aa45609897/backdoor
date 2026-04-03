import json
import sys


def analyze_json(data, indent=0, key_name="root"):
    """
    递归分析 JSON 结构
    """
    prefix = "  " * indent

    if isinstance(data, dict):
        print(f"{prefix}{key_name}: object {{")

        for k, v in data.items():
            analyze_json(v, indent + 1, k)

        print(f"{prefix}}}")

    elif isinstance(data, list):
        print(f"{prefix}{key_name}: array [")

        if len(data) > 0:
            print(f"{prefix}  (example element)")
            analyze_json(data[0], indent + 1, "[0]")
        else:
            print(f"{prefix}  (empty array)")

        print(f"{prefix}]")

    else:
        example = str(data)

        if len(example) > 60:
            example = example[:60] + "..."

        print(f"{prefix}{key_name}: {type(data).__name__}  (example: {example})")


def build_structure(data):
    """
    构建 JSON 结构 (schema)
    """
    if isinstance(data, dict):
        return {k: build_structure(v) for k, v in data.items()}

    elif isinstance(data, list):
        if len(data) == 0:
            return ["empty"]
        return [build_structure(data[0])]

    else:
        return type(data).__name__


def main():

    if len(sys.argv) < 2:
        print("Usage: python json_structure_analyzer.py file.json")
        return

    file_path = sys.argv[1]

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n===== JSON 层级示例 =====\n")
    analyze_json(data)

    print("\n===== JSON 结构 (Schema) =====\n")
    structure = build_structure(data)

    print(json.dumps(structure, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()