import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

class CLIPRunner:
    """
    CLIP 模型管理类：
    - 自动下载并缓存指定目录
    - 从本地加载模型
    - 计算图片与文本的相似度
    """
    def __init__(self, model_name='openai/clip-vit-base-patch32', cache_dir='./clip_cache', device=None):
        """
        Args:
            model_name (str): Hugging Face CLIP 模型名称
            cache_dir (str): 模型缓存目录
            device (str or None): 'cuda' 或 'cpu'，默认自动选择
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 自动选择设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型和 processor
        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_safetensors=True
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    def load_image(self, image_path):
        """
        读取图片并转换为 tensor
        """
        image = Image.open(image_path).convert("RGB")
        return image

    def compute_similarity(self, images, texts):
        """
        计算图像与文本的相似度
        Args:
            images (list of PIL.Image or str): 图片路径或 PIL.Image
            texts (list of str): 文本描述列表
        Returns:
            torch.Tensor: 相似度矩阵 [len(images), len(texts)]
        """
        # 如果传入的是路径，先加载图片
        images = [self.load_image(img) if isinstance(img, str) else img for img in images]

        # 处理输入
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)

        # 前向传播得到特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds  # [batch, dim]
            text_embeds = outputs.text_embeds    # [batch, dim]

        # L2 normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # 相似度矩阵
        similarity = image_embeds @ text_embeds.t()
        return similarity

    def predict(self, image_path, texts):
        """
        计算单张图片和多文本的相似度，并返回最匹配文本索引
        """
        sim = self.compute_similarity([image_path], texts)
        best_idx = sim.argmax(dim=-1).item()
        return best_idx, sim.cpu().numpy()

# ===================== 测试示例 =====================
if __name__ == "__main__":
    clip_runner = CLIPRunner(cache_dir="./model/clip/clip_cache")

    # 测试图片和文本
    test_image = "./model/test/test/1000268201.jpg"  # 替换为你的图片路径
    test_texts = [      
    "A CHILD IN A PINK DRESS IS CLIMBING UP A SET OF STAIRS IN AN ENTRY WAY.",
    "A LITTLE GIRL IN A PINK DRESS GOING INTO A WOODEN CABIN.",
    "A LITTLE GIRL CLIMBING THE STAIRS TO HER PLAYHOUSE.",
    "A LITTLE GIRL CLIMBING INTO A WOODEN PLAYHOUSE.",
    "A GIRL GOING INTO A WOODEN BUILDING."]

    best_idx, sim_matrix = clip_runner.predict(test_image, test_texts)
    print(f"最匹配文本索引: {best_idx}")
    print(f"相似度矩阵:\n{sim_matrix}")
    print(f"最匹配文本: {test_texts[best_idx]}")
