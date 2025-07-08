"""
增强型图像理解示例

演示如何使用MemoryX的增强型图像理解功能。
"""

import sys
import os
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memoryx.enhanced import EnhancedMemoryXImage

def create_test_image(color, size=(100, 100)):
    """创建测试图像"""
    return Image.new('RGB', size, color)

def main():
    # 初始化增强型图像记忆模块
    enhanced_image_memory = EnhancedMemoryXImage(user_id="user123")
    
    print("=" * 50)
    print("MemoryX增强型图像理解示例")
    print("=" * 50)
    
    # 创建测试图像
    red_image = create_test_image('red')
    blue_image = create_test_image('blue')
    green_image = create_test_image('green')
    
    # 添加增强型图像记忆
    print("\n添加红色图像（带增强分析）...")
    red_result = enhanced_image_memory.add_image_memory(
        image=red_image,
        description="这是一张红色的图片，红色代表热情和活力",
        importance=0.7
    )
    print(f"图像ID: {red_result['id']}")
    print(f"图像描述: {red_result['description']}")
    
    # 添加第二张图像
    print("\n添加蓝色图像（带增强分析）...")
    blue_result = enhanced_image_memory.add_image_memory(
        image=blue_image,
        description="这是一张蓝色的图片，蓝色代表平静和信任",
        importance=0.6
    )
    print(f"图像ID: {blue_result['id']}")
    print(f"图像描述: {blue_result['description']}")
    
    # 通过概念检索图像
    print("\n通过概念检索图像...")
    concept_results = enhanced_image_memory.retrieve_by_concept(
        concept_query="热情 活力",
        max_results=2
    )
    
    print("\n检索结果:")
    for i, result in enumerate(concept_results):
        print(f"{i+1}. 图像ID: {result['id']}")
        print(f"   描述: {result['metadata']['description']}")
        print(f"   相似度: {result['similarity']:.2f}")
    
    # 添加第三张图像
    print("\n添加绿色图像（带增强分析）...")
    green_result = enhanced_image_memory.add_image_memory(
        image=green_image,
        description="这是一张绿色的图片，绿色代表生机和和谐",
        importance=0.5
    )
    print(f"图像ID: {green_result['id']}")
    print(f"图像描述: {green_result['description']}")
    
    # 分析图像情感
    print("\n分析图像情感...")
    emotion_analysis = enhanced_image_memory.analyze_image_emotions(red_image)
    
    print("\n情感分析结果:")
    print(f"颜色分析: {emotion_analysis['color_analysis']}")
    print(f"情感预测: {emotion_analysis['emotions']}")
    print(f"美学评分: {emotion_analysis['aesthetic_score']}")
    
    # 查找图像关联
    print("\n查找图像关联...")
    connections = enhanced_image_memory.find_image_connections(
        image_id=red_result['id'],
        connection_type="semantic",
        depth=2
    )
    
    print("\n关联结果:")
    print(f"关联数量: {len(connections['connections'])}")
    print(f"关联叙述: {connections['narrative']}")
    
    # 生成图像故事
    print("\n生成图像故事...")
    image_ids = [red_result['id'], blue_result['id'], green_result['id']]
    story = enhanced_image_memory.generate_image_story(image_ids)
    
    print("\n图像故事:")
    print(f"叙事文本: {story['narrative_text']}")
    print(f"摘要: {story['summary']}")

if __name__ == "__main__":
    main()