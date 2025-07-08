"""
MemoryX 集成模块: 整合文本记忆和图像记忆功能
==============================================

本模块将MemoryX的文本记忆和图像记忆功能整合到一起，提供完整的多模态记忆系统:
1. 统一的记忆接口 - 同时处理文本和图像记忆
2. 跨模态检索 - 支持文本到图像、图像到文本的检索
3. 综合记忆管理 - 统一管理不同模态的记忆
4. 增强的对话体验 - 利用多模态记忆提供更丰富的交互
"""

import os
import time
import random
import numpy as np
from datetime import datetime
from PIL import Image

# 导入文本记忆模块
# 由于memoryx_advanced.py中没有直接导出MemoryController类
# 这里我们创建一个简化版本的文本记忆控制器
class AdvancedMemoryController:
    """简化版文本记忆控制器（用于演示）"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.memories = []
        self.memory_id = 0
    
    def add_interaction(self, user_input, assistant_response, conversation_id):
        """添加交互到记忆"""
        self.memory_id += 1
        memory = {
            'id': self.memory_id,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'conversation_id': conversation_id,
            'timestamp': time.time(),
            'importance': 0.6,
            'content': f"用户: {user_input}\n助手: {assistant_response}"
        }
        self.memories.append(memory)
        return memory
    
    def retrieve_memories(self, query, search_type="hybrid", top_k=3):
        """检索相关记忆"""
        # 简化版检索，仅基于简单的关键词匹配
        results = []
        for memory in self.memories:
            # 简单相似度计算（实际应使用嵌入和向量相似度）
            similarity = 0
            for word in query.lower().split():
                if word in memory['content'].lower():
                    similarity += 0.1
            
            if similarity > 0:
                results.append({
                    'id': memory['id'],
                    'content': memory['content'],
                    'similarity': similarity,
                    'metadata': {
                        'timestamp': memory['timestamp'],
                        'importance': memory['importance']
                    }
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回前K个结果
        return results[:top_k]

# 导入图像记忆模块
from memoryx_image import MemoryXImage, ImageProcessor

# ===============================
# 多模态记忆控制器
# ===============================

class MultimodalMemoryController:
    """多模态记忆控制器，整合文本和图像记忆"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        
        # 初始化文本记忆控制器
        self.text_memory = AdvancedMemoryController(user_id)
        
        # 初始化图像记忆控制器
        self.image_memory = MemoryXImage(user_id)
        
        # 工作记忆（对话上下文）
        self.working_memory = {}
        self.max_turns = 10
    
    def add_text_memory(self, text, metadata=None):
        """添加文本记忆"""
        return self.text_memory.add_interaction(
            user_input=text,
            assistant_response="",  # 这里可以留空，因为是用户提供的信息
            conversation_id="memory_input"
        )
    
    def add_image_memory(self, image, description=None, metadata=None):
        """添加图像记忆"""
        # 准备标签
        tags = []
        if metadata and 'tags' in metadata:
            tags = metadata['tags']
            
        # 准备重要性
        importance = 0.5
        if metadata and 'importance' in metadata:
            importance = metadata['importance']
            
        return self.image_memory.add_image_memory(
            image=image,
            description=description,
            importance=importance,
            tags=tags
        )
    
    def retrieve_by_text(self, text_query, max_text_results=3, max_image_results=2):
        """通过文本检索记忆（文本和图像）"""
        # 检索文本记忆
        text_memories = self.text_memory.retrieve_memories(
            query=text_query,
            search_type="hybrid",
            top_k=max_text_results
        )
        
        # 检索图像记忆
        image_memories = self.image_memory.retrieve_by_text(
            text_query=text_query,
            max_results=max_image_results
        )
        
        return {
            "text": text_memories,
            "image": image_memories
        }
    
    def retrieve_by_image(self, image_query, max_results=3):
        """通过图像检索记忆（主要是图像）"""
        return self.image_memory.retrieve_by_image(
            image_query=image_query,
            max_results=max_results
        )
    
    def update_working_memory(self, conversation_id, user_input, assistant_response, image=None):
        """更新工作记忆"""
        if conversation_id not in self.working_memory:
            self.working_memory[conversation_id] = []
            
        turn = {
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': time.time()
        }
        
        # 如果有图像，添加图像信息
        if image is not None:
            # 处理图像
            processor = ImageProcessor()
            # 生成描述
            description = processor.generate_description(image)
            
            # 添加到记忆
            image_result = self.add_image_memory(image)
            
            # 添加到对话轮次
            turn['image'] = {
                'id': image_result['id'],
                'description': description
            }
        
        self.working_memory[conversation_id].append(turn)
        
        # 保持对话历史在最大轮次以内
        if len(self.working_memory[conversation_id]) > self.max_turns:
            self.working_memory[conversation_id] = self.working_memory[conversation_id][-self.max_turns:]
    
    def get_working_memory(self, conversation_id):
        """获取工作记忆（对话上下文）"""
        if conversation_id not in self.working_memory:
            return ""
            
        context = ""
        for turn in self.working_memory[conversation_id]:
            # 添加用户输入
            context += f"用户: {turn['user']}\n"
            
            # 如果有图像，添加图像描述
            if 'image' in turn:
                context += f"[用户分享了图片: {turn['image']['description']}]\n"
                
            # 添加助手响应
            context += f"助手: {turn['assistant']}\n"
        
        return context
    
    def build_enhanced_prompt(self, user_input, current_context, relevant_memories, image=None):
        """构建增强提示"""
        prompt = "系统: 你是一个具有记忆能力的AI助手，能够记住文本和图像。请基于以下信息回答用户的问题。\n\n"
        
        # 如果用户提供了图像，添加图像上下文
        if image is not None:
            # 处理图像
            processor = ImageProcessor()
            # 生成描述
            description = processor.generate_description(image)
            prompt += f"[用户提供了一张图片: {description}]\n\n"
        
        # 添加相关图像记忆
        if relevant_memories and 'image' in relevant_memories and relevant_memories['image']:
            prompt += "相关图像记忆:\n"
            for i, memory in enumerate(relevant_memories['image']):
                prompt += f"[相关图像记忆 {i+1}: {memory['metadata']['description']}]\n"
            prompt += "\n"
        
        # 添加相关文本记忆
        if relevant_memories and 'text' in relevant_memories and relevant_memories['text']:
            prompt += "相关文本记忆:\n"
            for i, memory in enumerate(relevant_memories['text']):
                prompt += f"[相关文本记忆 {i+1}: {memory['content']}]\n"
            prompt += "\n"
        
        # 添加当前对话上下文
        if current_context:
            prompt += "当前对话:\n"
            prompt += current_context
            prompt += "\n"
        
        # 添加用户输入
        prompt += f"用户: {user_input}\n"
        prompt += "助手: "
        
        return prompt
    
    def optimize_storage(self):
        """优化存储空间"""
        # 优化图像存储
        self.image_memory.optimize_storage()
        
        # 优化文本存储（如果有实现）
        # self.text_memory.optimize_storage()
        
        return True

# ===============================
# 多模态记忆适配器
# ===============================

class MemoryXMultimodal:
    """多模态记忆适配器，集成到LLM系统"""
    
    def __init__(self, llm_provider, user_id):
        self.llm = llm_provider
        self.user_id = user_id
        
        # 初始化多模态记忆控制器
        self.memory_controller = MultimodalMemoryController(user_id)
        
        # 对话ID
        self.conversation_id = "demo_conversation"
    
    def generate_response(self, user_input, image=None):
        """生成增强记忆的响应，支持多模态输入"""
        print("\n处理用户输入:", user_input)
        
        # 处理图像输入
        if image is not None:
            print("处理图像输入...")
            processor = ImageProcessor()
            description = processor.generate_description(image)
            print(f"图像描述: {description}")
        
        # 获取当前对话上下文
        current_context = self.memory_controller.get_working_memory(self.conversation_id)
        print("\n当前对话上下文:", current_context if current_context else "[无上下文]")
        
        # 检索相关记忆
        relevant_memories = self.memory_controller.retrieve_by_text(
            text_query=user_input,
            max_text_results=3,
            max_image_results=2
        )
        
        # 打印检索到的记忆
        if relevant_memories['text']:
            print("\n检索到的相关文本记忆:")
            for i, memory in enumerate(relevant_memories['text']):
                print(f"{i+1}. {memory['content']}")
        
        if relevant_memories['image']:
            print("\n检索到的相关图像记忆:")
            for i, memory in enumerate(relevant_memories['image']):
                print(f"{i+1}. 图像ID: {memory['id']}, 描述: {memory['metadata']['description']}")
        
        # 构建增强提示
        enhanced_prompt = self.memory_controller.build_enhanced_prompt(
            user_input=user_input,
            current_context=current_context,
            relevant_memories=relevant_memories,
            image=image
        )
        
        print("\n增强提示:")
        print("-" * 60)
        print(enhanced_prompt)
        print("-" * 60)
        
        # 调用LLM生成响应
        response = self.llm(enhanced_prompt)
        print("\nLLM响应:", response)
        
        # 更新记忆
        self.memory_controller.update_working_memory(
            conversation_id=self.conversation_id,
            user_input=user_input,
            assistant_response=response,
            image=image
        )
        
        # 定期优化存储
        if random.random() < 0.1:  # 10%的概率执行优化
            self.memory_controller.optimize_storage()
        
        return response

# ===============================
# 模拟LLM响应函数
# ===============================

def simulate_llm_response(prompt):
    """模拟LLM响应（演示用）"""
    # 检查提示中是否包含图像相关内容
    has_image = "[用户提供了一张图片" in prompt or "[相关图像记忆" in prompt or "[用户分享了图片" in prompt
    
    # 检查图像颜色
    has_red = "红色" in prompt
    has_blue = "蓝色" in prompt
    has_green = "绿色" in prompt
    
    # 根据提示内容生成响应
    if "红色" in prompt and "图片" in prompt and ("[用户提供了一张图片" in prompt or "[用户分享了图片" in prompt):
        return "我看到您分享了一张红色的图片。红色通常象征热情、活力和力量。您想分享更多关于这张图片的信息吗？"
    elif "蓝色" in prompt and "图片" in prompt and ("[用户提供了一张图片" in prompt or "[用户分享了图片" in prompt):
        return "我看到您分享了一张蓝色的图片。蓝色常常给人平静、稳定和信任的感觉。这张图片有什么特别的意义吗？"
    elif "绿色" in prompt and "图片" in prompt and ("[用户提供了一张图片" in prompt or "[用户分享了图片" in prompt):
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

def run_multimodal_memory_demo():
    """运行多模态记忆演示"""
    print("=" * 80)
    print("MemoryX多模态记忆模块演示")
    print("=" * 80)
    
    # 初始化适配器
    adapter = MemoryXMultimodal(simulate_llm_response, "demo_user")
    
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
        {"text": "你好，我想分享一些图片和文字。", "image": None},
        {"text": "这是一张红色的图片，红色代表热情。", "image": red_image},
        {"text": "你能记住这张图片吗？", "image": None},
        {"text": "这是一张蓝色的图片，蓝色代表平静。", "image": blue_image},
        {"text": "你还记得我之前分享的红色图片吗？", "image": None},
        {"text": "这是一张绿色的图片，绿色代表生机。", "image": green_image},
        {"text": "总结一下我分享过的所有图片和它们的含义。", "image": None}
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
    # 注意：这个演示需要先运行memoryx_advanced.py和memoryx_image.py
    # 实际运行时可能需要调整导入和依赖关系
    run_multimodal_memory_demo()