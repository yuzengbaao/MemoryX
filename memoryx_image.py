"""
MemoryX 图像记忆模块: 为MemoryX增加图像记忆功能
==============================================

本模块扩展了MemoryX的功能，增加了图像记忆能力:
1. 图像处理与嵌入 - 使用视觉模型生成图像表示
2. 图像-文本跨模态检索 - 支持图像与文本之间的双向检索
3. 图像描述生成 - 自动为图像生成文本描述
4. 图像记忆管理 - 图像压缩、存储和检索优化
"""

import os
import io
import time
import math
import json
import base64
import hashlib
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Any

# 模拟依赖，实际应用中应使用真实的库
# 如 transformers, torch, clip 等
class SimulatedDependencies:
    """模拟依赖库，实际应用中应替换为真实库"""
    
    @staticmethod
    def resize_image(image, size=(224, 224)):
        """调整图像大小"""
        if isinstance(image, str):
            # 如果是文件路径
            image = Image.open(image)
        elif isinstance(image, bytes):
            # 如果是字节数据
            image = Image.open(io.BytesIO(image))
        
        return image.resize(size)
    
    @staticmethod
    def encode_image(image):
        """将图像编码为向量（模拟）"""
        # 实际应使用如CLIP、ResNet等模型
        # 这里仅生成随机向量作为示例
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        np.random.seed(int(image_hash[:8], 16))
        return np.random.random(512)  # 512维向量
    
    @staticmethod
    def generate_image_description(image):
        """为图像生成描述（模拟）"""
        # 实际应使用如BLIP、OFA等图像描述模型
        # 这里根据图像颜色生成简单描述
        
        # 获取图像主要颜色
        if isinstance(image, Image.Image):
            # 计算平均颜色
            img_array = np.array(image)
            avg_color = img_array.mean(axis=(0, 1))
            
            # 确定主要颜色
            r, g, b = avg_color if len(avg_color) >= 3 else (avg_color[0], avg_color[0], avg_color[0])
            
            # 简单颜色判断
            if r > max(g, b) + 50:
                return "这是一张以红色为主的图片。"
            elif g > max(r, b) + 50:
                return "这是一张以绿色为主的图片。"
            elif b > max(r, g) + 50:
                return "这是一张以蓝色为主的图片。"
            else:
                return "这是一张包含多种颜色的图片。"
        
        return "这是一张图片，包含了一些视觉内容。"
    
    @staticmethod
    def compress_image(image, quality=85):
        """压缩图像（模拟）"""
        # 实际应根据需要调整压缩参数
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality)
        return output.getvalue()
    
    @staticmethod
    def image_to_base64(image):
        """将图像转换为base64编码"""
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        else:
            raise ValueError("不支持的图像类型")
    
    @staticmethod
    def base64_to_image(base64_str):
        """将base64编码转换为图像"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))

# ===============================
# 图像处理与嵌入
# ===============================

class ImageProcessor:
    """图像处理器，负责图像预处理和嵌入生成"""
    
    def __init__(self, model_name="simulated"):
        self.model_name = model_name
        self.image_size = (224, 224)  # 默认图像大小
        self.embedding_dim = 512  # 默认嵌入维度
        
        # 在实际应用中，这里应加载真实的视觉模型
        self.sim = SimulatedDependencies()
    
    def preprocess(self, image_input):
        """预处理图像"""
        # 处理不同类型的输入
        if isinstance(image_input, str):
            # 文件路径
            if os.path.exists(image_input):
                image = Image.open(image_input)
            else:
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
        elif isinstance(image_input, bytes):
            # 字节数据
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            # PIL图像对象
            image = image_input
        else:
            raise TypeError(f"不支持的图像输入类型: {type(image_input)}")
        
        # 调整大小
        resized_image = self.sim.resize_image(image, self.image_size)
        
        return resized_image
    
    def generate_embedding(self, image):
        """生成图像嵌入"""
        # 预处理图像
        processed_image = self.preprocess(image)
        
        # 生成嵌入
        embedding = self.sim.encode_image(processed_image)
        
        return embedding
    
    def generate_description(self, image):
        """为图像生成文本描述"""
        # 预处理图像
        processed_image = self.preprocess(image)
        
        # 生成描述
        description = self.sim.generate_image_description(processed_image)
        
        return description
    
    def compress_image(self, image, level=1):
        """压缩图像
        
        级别:
        1 = 轻度压缩 (质量85)
        2 = 中度压缩 (质量65)
        3 = 高度压缩 (质量45, 调整大小)
        """
        # 预处理图像
        processed_image = self.preprocess(image)
        
        # 根据级别确定压缩参数
        if level == 1:
            quality = 85
            resize_factor = 1.0
        elif level == 2:
            quality = 65
            resize_factor = 0.8
        else:  # level 3
            quality = 45
            resize_factor = 0.5
        
        # 调整大小
        if resize_factor < 1.0:
            new_size = (int(processed_image.width * resize_factor), 
                       int(processed_image.height * resize_factor))
            processed_image = processed_image.resize(new_size)
        
        # 压缩
        compressed_data = self.sim.compress_image(processed_image, quality)
        
        return compressed_data
    
    def image_to_base64(self, image):
        """将图像转换为base64编码"""
        return self.sim.image_to_base64(image)
    
    def base64_to_image(self, base64_str):
        """将base64编码转换为图像"""
        return self.sim.base64_to_image(base64_str)

# ===============================
# 图像记忆存储
# ===============================

class ImageMemoryStore:
    """图像记忆存储，管理图像数据和元数据"""
    
    def __init__(self, base_dir="/tmp/memoryx_images"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 图像元数据索引
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # 图像处理器
        self.processor = ImageProcessor()
    
    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_image(self, image, metadata=None):
        """添加图像到存储"""
        # 生成唯一ID
        image_id = f"img_{int(time.time())}_{len(self.metadata) + 1}"
        
        # 处理图像
        processed_image = self.processor.preprocess(image)
        
        # 生成嵌入
        embedding = self.processor.generate_embedding(processed_image)
        
        # 生成描述
        description = self.processor.generate_description(processed_image)
        
        # 压缩并存储图像
        compressed_image = self.processor.compress_image(processed_image, level=1)
        image_path = self.base_dir / f"{image_id}.jpg"
        with open(image_path, 'wb') as f:
            f.write(compressed_image)
        
        # 存储元数据
        self.metadata[image_id] = {
            'path': str(image_path),
            'embedding': embedding.tolist(),
            'description': description,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'access_count': 0,
            'compression_level': 1,
            **(metadata or {})
        }
        
        # 保存元数据
        self._save_metadata()
        
        return {
            'id': image_id,
            'description': description,
            'metadata': self.metadata[image_id]
        }
    
    def get_image(self, image_id):
        """获取图像"""
        if image_id not in self.metadata:
            raise ValueError(f"图像不存在: {image_id}")
        
        # 更新访问统计
        self.metadata[image_id]['last_accessed'] = time.time()
        self.metadata[image_id]['access_count'] += 1
        self._save_metadata()
        
        # 读取图像
        image_path = self.metadata[image_id]['path']
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        return {
            'id': image_id,
            'data': image_data,
            'description': self.metadata[image_id]['description'],
            'metadata': self.metadata[image_id]
        }
    
    def get_embedding(self, image_id):
        """获取图像嵌入"""
        if image_id not in self.metadata:
            raise ValueError(f"图像不存在: {image_id}")
        
        return np.array(self.metadata[image_id]['embedding'])
    
    def search_similar(self, query_embedding, top_k=5):
        """搜索相似图像"""
        if len(self.metadata) == 0:
            return []
        
        # 计算相似度
        results = []
        for image_id, data in self.metadata.items():
            embedding = np.array(data['embedding'])
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            results.append((image_id, similarity, data))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前K个结果
        return [
            {'id': image_id, 'similarity': similarity, 'metadata': data}
            for image_id, similarity, data in results[:top_k]
        ]
    
    def compress_image(self, image_id, level):
        """压缩图像"""
        if image_id not in self.metadata:
            raise ValueError(f"图像不存在: {image_id}")
        
        # 读取原始图像
        image_path = self.metadata[image_id]['path']
        image = Image.open(image_path)
        
        # 压缩图像
        compressed_data = self.processor.compress_image(image, level)
        
        # 保存压缩后的图像
        with open(image_path, 'wb') as f:
            f.write(compressed_data)
        
        # 更新元数据
        self.metadata[image_id]['compression_level'] = level
        self._save_metadata()
        
        return True
    
    def get_all_images(self):
        """获取所有图像元数据"""
        return [
            {'id': image_id, 'metadata': data}
            for image_id, data in self.metadata.items()
        ]

# ===============================
# 跨模态记忆检索
# ===============================

class CrossModalRetriever:
    """跨模态检索器，支持图像-文本双向检索"""
    
    def __init__(self, image_store, text_embedder=None):
        self.image_store = image_store
        self.image_processor = ImageProcessor()
        
        # 文本嵌入器（实际应用中应使用真实的文本嵌入模型）
        self.text_embedder = text_embedder or self._default_text_embedder
    
    def _default_text_embedder(self, text):
        """默认文本嵌入器（模拟）"""
        # 实际应使用如Sentence-BERT等模型
        # 这里仅生成随机向量作为示例
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        return np.random.random(512)  # 512维向量，与图像嵌入维度相同
    
    def text_to_image(self, text_query, top_k=5):
        """文本到图像检索"""
        # 生成文本嵌入
        text_embedding = self.text_embedder(text_query)
        
        # 搜索相似图像
        results = self.image_store.search_similar(text_embedding, top_k)
        
        return results
    
    def image_to_text(self, image_query, text_corpus, top_k=5):
        """图像到文本检索"""
        # 生成图像嵌入
        image_embedding = self.image_processor.generate_embedding(image_query)
        
        # 计算与文本的相似度
        results = []
        for text_id, text in text_corpus.items():
            text_embedding = self.text_embedder(text)
            similarity = np.dot(image_embedding, text_embedding) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding))
            results.append((text_id, text, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前K个结果
        return [
            {'id': text_id, 'text': text, 'similarity': similarity}
            for text_id, text, similarity in results[:top_k]
        ]
    
    def image_to_image(self, image_query, top_k=5):
        """图像到图像检索"""
        # 生成图像嵌入
        image_embedding = self.image_processor.generate_embedding(image_query)
        
        # 搜索相似图像
        results = self.image_store.search_similar(image_embedding, top_k)
        
        return results

# ===============================
# 图像记忆管理器
# ===============================

class ImageMemoryManager:
    """图像记忆管理器，负责图像记忆的管理和优化"""
    
    def __init__(self, image_store):
        self.image_store = image_store
    
    def evaluate_importance(self, image_id):
        """评估图像重要性"""
        if image_id not in self.image_store.metadata:
            raise ValueError(f"图像不存在: {image_id}")
        
        metadata = self.image_store.metadata[image_id]
        
        # 基础重要性（如果元数据中有提供）
        importance = metadata.get('importance', 0.5)
        
        # 访问频率因子
        access_count = metadata.get('access_count', 0)
        frequency_factor = min(0.3, 0.05 * math.log(1 + access_count))
        
        # 时间衰减因子
        time_elapsed = time.time() - metadata.get('created_at', time.time())
        time_factor = math.exp(-0.01 * time_elapsed / (24 * 3600))  # 每天衰减约1%
        
        # 综合评分
        final_score = importance + frequency_factor + time_factor * 0.2
        
        return max(0.1, min(1.0, final_score))
    
    def determine_compression_level(self, importance_score):
        """根据重要性确定压缩级别"""
        if importance_score > 0.7:
            return 1  # 轻度压缩
        elif importance_score > 0.4:
            return 2  # 中度压缩
        else:
            return 3  # 高度压缩
    
    def optimize_storage(self):
        """优化存储空间"""
        # 获取所有图像
        all_images = self.image_store.get_all_images()
        
        # 评估每个图像的重要性并决定压缩级别
        for image_data in all_images:
            image_id = image_data['id']
            
            # 评估重要性
            importance = self.evaluate_importance(image_id)
            
            # 确定压缩级别
            current_level = image_data['metadata'].get('compression_level', 1)
            target_level = self.determine_compression_level(importance)
            
            # 如果需要更高压缩
            if target_level > current_level:
                print(f"压缩图像 {image_id}: 重要性 {importance:.2f}, 压缩级别 {current_level} -> {target_level}")
                self.image_store.compress_image(image_id, target_level)
        
        return True

# ===============================
# MemoryX图像记忆集成
# ===============================

class MemoryXImage:
    """MemoryX图像记忆模块，集成到MemoryX系统"""
    
    def __init__(self, user_id, base_dir="/tmp/memoryx_images"):
        self.user_id = user_id
        self.user_dir = os.path.join(base_dir, user_id)
        
        # 初始化组件
        self.image_store = ImageMemoryStore(self.user_dir)
        self.image_processor = ImageProcessor()
        self.cross_modal_retriever = CrossModalRetriever(self.image_store)
        self.memory_manager = ImageMemoryManager(self.image_store)
    
    def add_image_memory(self, image, description=None, importance=0.5, tags=None):
        """添加图像记忆"""
        # 准备元数据
        metadata = {
            'user_id': self.user_id,
            'importance': importance,
            'tags': tags or [],
            'custom_description': description
        }
        
        # 添加图像
        result = self.image_store.add_image(image, metadata)
        
        # 如果提供了自定义描述，更新描述
        if description:
            self.image_store.metadata[result['id']]['description'] = description
            self.image_store._save_metadata()
        
        return result
    
    def retrieve_by_text(self, text_query, max_results=5):
        """通过文本检索图像"""
        return self.cross_modal_retriever.text_to_image(text_query, max_results)
    
    def retrieve_by_image(self, image_query, max_results=5):
        """通过图像检索相似图像"""
        return self.cross_modal_retriever.image_to_image(image_query, max_results)
    
    def get_image(self, image_id):
        """获取图像"""
        return self.image_store.get_image(image_id)
    
    def optimize_storage(self):
        """优化存储空间"""
        return self.memory_manager.optimize_storage()
    
    def get_all_images(self):
        """获取所有图像"""
        return self.image_store.get_all_images()
    
    def image_to_base64(self, image):
        """将图像转换为base64编码"""
        return self.image_processor.image_to_base64(image)
    
    def base64_to_image(self, base64_str):
        """将base64编码转换为图像"""
        return self.image_processor.base64_to_image(base64_str)

# ===============================
# 与MemoryX集成的适配器
# ===============================

class MemoryXWithImages:
    """集成图像记忆功能的MemoryX适配器"""
    
    def __init__(self, llm_provider, user_id):
        self.llm = llm_provider
        self.user_id = user_id
        
        # 初始化文本记忆控制器（假设已有）
        # 实际应导入之前实现的MemoryController
        self.text_memory = None  # 应替换为实际的MemoryController实例
        
        # 初始化图像记忆模块
        self.image_memory = MemoryXImage(user_id)
        
        # 对话ID
        self.conversation_id = "demo_conversation"
    
    def generate_response(self, user_input, image=None):
        """生成增强记忆的响应，支持图像输入"""
        print("\n处理用户输入:", user_input)
        
        # 处理图像输入
        image_context = ""
        if image is not None:
            print("处理图像输入...")
            
            # 添加图像到记忆
            image_result = self.image_memory.add_image_memory(
                image=image,
                importance=0.7  # 用户主动提供的图像通常较重要
            )
            
            # 获取图像描述
            image_description = image_result['description']
            image_context = f"[用户提供了一张图片: {image_description}]\n"
            
            print(f"图像已添加到记忆，ID: {image_result['id']}")
            print(f"图像描述: {image_description}")
        
        # 检索相关图像记忆
        image_memories = []
        if user_input:
            image_results = self.image_memory.retrieve_by_text(user_input, max_results=2)
            
            if image_results:
                print("\n检索到的相关图像记忆:")
                for i, result in enumerate(image_results):
                    print(f"{i+1}. 图像ID: {result['id']}, 相似度: {result['similarity']:.2f}")
                    print(f"   描述: {result['metadata']['description']}")
                    
                    # 添加到上下文
                    image_memories.append(
                        f"[相关图像记忆 {i+1}: {result['metadata']['description']}]"
                    )
        
        # 构建增强提示
        image_memory_context = "\n".join(image_memories)
        
        # 这里应集成文本记忆检索
        # 实际应使用之前实现的记忆控制器检索文本记忆
        text_memories = []
        text_memory_context = "\n".join(text_memories)
        
        # 构建完整提示
        enhanced_prompt = f"""系统: 你是一个具有记忆能力的AI助手，能够记住文本和图像。请基于以下信息回答用户的问题。

{image_context}

相关图像记忆:
{image_memory_context}

相关文本记忆:
{text_memory_context}

用户: {user_input}
助手: """
        
        print("\n增强提示:")
        print("-" * 60)
        print(enhanced_prompt)
        print("-" * 60)
        
        # 调用LLM生成响应
        response = self.llm(enhanced_prompt)
        print("\nLLM响应:", response)
        
        # 更新文本记忆（如果有）
        # 实际应使用之前实现的记忆控制器更新文本记忆
        
        # 定期优化图像存储
        if random.random() < 0.1:  # 10%的概率执行优化
            self.image_memory.optimize_storage()
        
        return response

# ===============================
# 模拟LLM响应函数
# ===============================

def simulate_llm_response(prompt):
    """模拟LLM响应（演示用）"""
    # 检查提示中是否包含图像相关内容
    has_image = "[用户提供了一张图片" in prompt or "[相关图像记忆" in prompt
    
    # 检查图像颜色
    has_red = "红色" in prompt
    has_blue = "蓝色" in prompt
    has_green = "绿色" in prompt
    
    # 根据提示内容生成响应
    if "红色" in prompt and "图片" in prompt and "[用户提供了一张图片" in prompt:
        return "我看到您分享了一张红色的图片。红色通常象征热情、活力和力量。您想分享更多关于这张图片的信息吗？"
    elif "蓝色" in prompt and "图片" in prompt and "[用户提供了一张图片" in prompt:
        return "我看到您分享了一张蓝色的图片。蓝色常常给人平静、稳定和信任的感觉。这张图片有什么特别的意义吗？"
    elif "绿色" in prompt and "图片" in prompt and "[用户提供了一张图片" in prompt:
        return "我看到您分享了一张绿色的图片。绿色通常代表自然、生长和和谐。您对这种颜色有特别的喜好吗？"
    elif has_image and "记得" in prompt and "红色" in prompt:
        return "是的，我记得您之前分享过一张红色的图片。红色是一种非常醒目的颜色，给人留下深刻印象。"
    elif "总结" in prompt and has_image:
        colors = []
        if "红色" in prompt or any("红色" in memory for memory in prompt.split("[相关图像记忆") if len(memory) > 5):
            colors.append("红色")
        if "蓝色" in prompt or any("蓝色" in memory for memory in prompt.split("[相关图像记忆") if len(memory) > 5):
            colors.append("蓝色")
        if "绿色" in prompt or any("绿色" in memory for memory in prompt.split("[相关图像记忆") if len(memory) > 5):
            colors.append("绿色")
            
        if colors:
            color_text = "、".join(colors)
            return f"根据我们的对话，您分享了几张不同颜色的图片，包括{color_text}图片。这些不同颜色可能代表不同的情绪和象征意义。"
        else:
            return "根据我们的对话，您分享了几张不同颜色的图片。每种颜色都有其独特的视觉效果和象征意义。"
    else:
        return "我理解您的问题。如果您想分享更多图片或有其他问题，请随时告诉我。"

# ===============================
# 演示函数
# ===============================

def run_image_memory_demo():
    """运行MemoryX图像记忆演示"""
    print("=" * 80)
    print("MemoryX图像记忆模块演示")
    print("=" * 80)
    
    # 初始化适配器
    adapter = MemoryXWithImages(simulate_llm_response, "demo_user")
    
    # 创建测试图像（纯色图像）
    def create_test_image(color, size=(100, 100)):
        image = Image.new('RGB', size, color)
        return image
    
    # 测试图像
    red_image = create_test_image('red')
    blue_image = create_test_image('blue')
    green_image = create_test_image('green')
    
    # 模拟对话
    conversations = [
        {"text": "你好，我想分享一些图片。", "image": None},
        {"text": "这是一张红色的图片。", "image": red_image},
        {"text": "你能记住这张图片吗？", "image": None},
        {"text": "这是一张蓝色的图片。", "image": blue_image},
        {"text": "你还记得我之前分享的红色图片吗？", "image": None},
        {"text": "这是一张绿色的图片。", "image": green_image},
        {"text": "总结一下我分享过的所有图片。", "image": None}
    ]
    
    for i, conv in enumerate(conversations):
        print("\n" + "=" * 80)
        print(f"对话轮次 {i+1}/{len(conversations)}")
        print("=" * 80)
        
        response = adapter.generate_response(conv["text"], conv["image"])
        
        print("\n最终响应:", response)
        
        # 模拟时间流逝
        if i < len(conversations) - 1:
            time_skip = random.randint(1, 5)
            print(f"\n[时间流逝: {time_skip} 小时后...]")

if __name__ == "__main__":
    run_image_memory_demo()