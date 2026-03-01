import os
# ===================== 配置中国镜像 =====================
# 设置Hugging Face镜像（支持多个可选镜像）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 推荐

import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration
)

class BLIP2Runner:
    """
    BLIP-2 模型管理类：
    - 自动下载并缓存指定目录
    - 从本地加载模型
    - 计算图片与多个文本的匹配程度（基于生成 loss）
    - 生成图片描述（新增功能）
    """

    def __init__(
        self,
        model_name="Salesforce/blip2-flan-t5-xl",
        cache_dir="./blip2_cache",
        device=None,
        dtype=torch.float16
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 自动选择设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # processor
        self.processor = Blip2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None
        )

        self.model.eval()

    def load_image(self, image_path):
        """加载图片"""
        image = Image.open(image_path).convert("RGB")
        return image

    @torch.no_grad()
    def compute_matching_scores(self, image, texts):
        """
        通过 conditional generation loss 计算 image-text 匹配程度
        score 越小，匹配度越高
        """
        scores = []

        for text in texts:
            prompt = f"Question: Does the image match the description: '{text}'? Answer:"

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            labels = self.processor.tokenizer(
                "yes",
                return_tensors="pt"
            ).input_ids.to(self.device)

            outputs = self.model(
                **inputs,
                labels=labels
            )

            loss = outputs.loss.item()
            scores.append(loss)

        return torch.tensor(scores)

    @torch.no_grad()
    def generate_caption(self, image, prompt=None, max_length=50, num_beams=5):
        """
        生成图片描述
        
        Args:
            image: PIL.Image 或图片路径
            prompt: 可选，引导生成的提示词（例如："A photo of"）
            max_length: 生成文本的最大长度
            num_beams: beam search 数量
        
        Returns:
            str: 生成的图片描述
        """
        # 如果传入的是路径，加载图片
        if isinstance(image, str):
            image = self.load_image(image)
        
        # 构建输入
        if prompt:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            # 如果不提供prompt，使用默认的图片描述生成
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

        # 生成描述
        generated_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,  # 为了确定性结果，使用beam search
            early_stopping=True
        )

        # 解码生成的文本
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()

        return generated_text
    
    @torch.no_grad()
    def generate_caption_with_prompts(self, image, prompts=None, **generate_kwargs):
        """
        使用不同的提示词生成多个描述
        
        Args:
            image: PIL.Image 或图片路径
            prompts: 提示词列表，如果为None则使用默认提示词
            **generate_kwargs: 传递给generate_caption的其他参数
        
        Returns:
            list: 生成的描述列表
        """
        if prompts is None:
            prompts = [
                "",  # 空提示词
                "A photo of",
                "A realistic photo of",
                "A detailed image of",
                "An image showing"
            ]
        
        captions = []
        for prompt in prompts:
            caption = self.generate_caption(image, prompt=prompt, **generate_kwargs)
            captions.append(caption)
        
        return captions

    def predict(self, image_path, texts):
        """
        返回最匹配文本索引和所有 score
        """
        image = self.load_image(image_path)
        scores = self.compute_matching_scores(image, texts)
        best_idx = scores.argmin().item()
        return best_idx, scores.cpu().numpy()


# ===================== 测试示例 =====================
if __name__ == "__main__":
    blip2_runner = BLIP2Runner(cache_dir="./model/blip/blip2_cache")

    test_image = "./model/test/test/1000268201.jpg"
    
    # 测试1: 生成图片描述
    print("测试1: 生成图片描述")
    caption = blip2_runner.generate_caption(test_image)
    print(f"生成的描述: {caption}")
    
    # 测试2: 使用提示词生成描述
    print("\n测试2: 使用提示词生成描述")
    caption_with_prompt = blip2_runner.generate_caption(
        test_image, 
        prompt="A realistic photo of"
    )
    print(f"使用提示词生成的描述: {caption_with_prompt}")
    
    # 测试3: 使用多个提示词生成描述
    print("\n测试3: 使用多个提示词生成描述")
    prompts = ["", "A photo of", "An image showing a"]
    multi_captions = blip2_runner.generate_caption_with_prompts(
        test_image, 
        prompts=prompts
    )
    for i, (prompt, cap) in enumerate(zip(prompts, multi_captions)):
        print(f"提示词 '{prompt}': {cap}")
    
    # 测试4: 原有匹配功能（保持不变）
    print("\n测试4: 文本匹配功能")
    test_texts = [
        "A child in a pink dress is climbing stairs.",
        "A dog is running on the beach.",
        "A car is parked on the street.",
        "A little girl climbing into a wooden playhouse.",
        "A man riding a bicycle."
    ]

    best_idx, scores = blip2_runner.predict(test_image, test_texts)

    print("匹配分数（越小越匹配）:")
    for i, s in enumerate(scores):
        print(f"[{i}] {s:.4f} | {test_texts[i]}")

    print(f"\n最匹配文本索引: {best_idx}")
    print(f"最匹配文本: {test_texts[best_idx]}")
    
    # 测试5: 完整工作流程示例
    print("\n测试5: 完整工作流程示例")
    # 先生成描述
    auto_caption = blip2_runner.generate_caption(test_image)
    print(f"模型自动生成的描述: {auto_caption}")
    
    # 然后与候选文本进行匹配
    candidate_texts = [auto_caption] + test_texts
    best_idx, scores = blip2_runner.predict(test_image, candidate_texts)
    print(f"\n模型生成描述与图片的匹配分数: {scores[0]:.4f}")