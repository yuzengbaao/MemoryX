"""
图像记忆示例

演示如何使用MemoryX的图像记忆功能。
"""

import sys
import os
import time
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memoryx import MemoryXImage

def create_test_image(color, size=(100, 100)):
    """创建测试图像"""
    return Image.new('RGB', size, color)

def main():
    # 初始化图像记忆模块
    image_memory = MemoryXImage(user_id="user123")
    
    print("=" * 50)
    print("MemoryX图像记忆示例")
    print("=" * 50)
    
    # 创建测试图像
    red_image = create_test_image('red')
    blue_image = create_test_image('blue')
    green_image = create_test_image('green')
    
    # 添加图像记忆
    print("\n添加红色图像...")
    red_result = image_memory.add_image_memory(
        image=red_image,
        description="这是一张红色的图片，红色代表热情",
        importance=0.7
    )
    print(f"图像ID: {red_result['id']}")
    print(f"图像描述: {red_result['description']}")
    
    # 添加第二张图像
    print("\n添加蓝色图像...")
    blue_result = image_memory.add_image_memory(
        image=blue_image,
        description="这是一张蓝色的图片，蓝色代表平静",
        importance=0.6
    )
    print(f"图像ID: {blue_result['id']}")
    print(f"图像描述: {blue_result['description']}")
    
    # 通过文本检索图像
    print("\n通过文本检索图像...")
    image_results = image_memory.retrieve_by_text(
        text_query="红色图片",
        max_results=2
    )
    
    print("\n检索结果:")
    for i, result in enumerate(image_results):
        print(f"{i+1}. 图像ID: {result['id']}")
        print(f"   描述: {result['metadata']['description']}")
        print(f"   相似度: {result['similarity']:.2f}")
    
    # 添加第三张图像
    print("\n添加绿色图像...")
    green_result = image_memory.add_image_memory(
        image=green_image,
        description="这是一张绿色的图片，绿色代表生机",
        importance=0.5
    )
    print(f"图像ID: {green_result['id']}")
    print(f"图像描述: {green_result['description']}")
    
    # 通过图像检索相似图像
    print("\n通过图像检索相似图像...")
    similar_results = image_memory.retrieve_by_image(
        image_query=green_image,
        max_results=3
    )
    
    print("\n检索结果:")
    for i, result in enumerate(similar_results):
        print(f"{i+1}. 图像ID: {result['id']}")
        print(f"   描述: {result['metadata']['description']}")
        print(f"   相似度: {result['similarity']:.2f}")
    
    # 优化存储
    print("\n优化存储...")
    image_memory.optimize_storage()
    
    # 获取所有图像
    print("\n获取所有图像...")
    all_images = image_memory.get_all_images()
    
    print(f"\n共有 {len(all_images)} 张图像:")
    for i, image_data in enumerate(all_images):
        print(f"{i+1}. 图像ID: {image_data['id']}")
        print(f"   描述: {image_data['metadata']['description']}")
        print(f"   压缩级别: {image_data['metadata']['compression_level']}")

if __name__ == "__main__":
    main()