"""
基础文本记忆示例

演示如何使用MemoryX的基础文本记忆功能。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memoryx import MemoryController

def main():
    # 初始化记忆控制器
    memory_controller = MemoryController(user_id="user123")
    
    print("=" * 50)
    print("MemoryX基础文本记忆示例")
    print("=" * 50)
    
    # 添加记忆
    print("\n添加记忆...")
    memory_controller.add_interaction(
        user_input="我的名字是张三",
        assistant_response="很高兴认识你，张三",
        conversation_id="conversation1"
    )
    
    memory_controller.add_interaction(
        user_input="我住在北京",
        assistant_response="北京是一个美丽的城市",
        conversation_id="conversation1"
    )
    
    memory_controller.add_interaction(
        user_input="我喜欢阅读和旅行",
        assistant_response="阅读和旅行都是很好的爱好",
        conversation_id="conversation1"
    )
    
    # 检索记忆
    print("\n检索记忆...")
    memories = memory_controller.retrieve_memories(
        query="我的名字是什么",
        search_type="semantic",
        top_k=3
    )
    
    print("\n检索结果:")
    for i, memory in enumerate(memories):
        print(f"{i+1}. {memory['content']}")
    
    # 添加更多记忆
    print("\n添加更多记忆...")
    memory_controller.add_interaction(
        user_input="我今年35岁",
        assistant_response="35岁是一个很好的年龄",
        conversation_id="conversation1"
    )
    
    memory_controller.add_interaction(
        user_input="我是一名软件工程师",
        assistant_response="软件工程是一个很有前途的职业",
        conversation_id="conversation1"
    )
    
    # 再次检索记忆
    print("\n再次检索记忆...")
    memories = memory_controller.retrieve_memories(
        query="我的职业是什么",
        search_type="semantic",
        top_k=3
    )
    
    print("\n检索结果:")
    for i, memory in enumerate(memories):
        print(f"{i+1}. {memory['content']}")
    
    # 总结记忆
    print("\n总结所有记忆...")
    all_memories = memory_controller.retrieve_memories(
        query="总结用户信息",
        search_type="hybrid",
        top_k=5
    )
    
    print("\n用户信息总结:")
    for i, memory in enumerate(all_memories):
        print(f"{i+1}. {memory['content']}")

if __name__ == "__main__":
    main()