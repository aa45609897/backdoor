import open_clip

# ===============================
# 模型配置
# ===============================
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
MODEL_SAVE_DIR = "./model/clip/openclip_local"

# 加载模型
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device="cpu",           # 可以改成 "cuda" 如果有GPU
    cache_dir=MODEL_SAVE_DIR
)

# ===============================
# 打印所有层名字
# ===============================
def print_all_layers(model):
    print("\n===== 所有层名 =====")
    for name, module in model.named_modules():
        print(name)

# ===============================
# 分别打印视觉编码器和文本编码器的层
# ===============================
def print_visual_text_layers(model):
    print("\n===== 视觉编码器层 =====")
    for name, _ in model.visual.named_modules():
        print(name)

    print("\n===== 文本编码器层 =====")
    for name, _ in model.transformer.named_modules():
        print(name)

# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    print_all_layers(model)
    print_visual_text_layers(model)
