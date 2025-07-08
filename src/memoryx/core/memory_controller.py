"""
MemoryX: 通用LLM记忆增强模块演示
=================================

这个演示展示了MemoryX模块的核心功能，包括：
1. 记忆存储与检索
2. 动态记忆强化
3. 策略性遗忘
4. 与LLM的集成

注意：这是一个简化的演示实现，实际应用中需要更完善的错误处理和优化。
"""

import os
import math
import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity

# 模拟向量嵌入函数（实际应用中应使用真实的嵌入模型）
def embed_text(text):
    """将文本转换为向量嵌入（演示用）"""
    # 简化的嵌入模拟，实际应用中应使用如OpenAI的嵌入API或本地模型
    random.seed(hash(text) % 10000)
    return np.array([random.random() for _ in range(384)])

# 模拟LLM生成函数（实际应用中应集成真实的LLM API）
def simulate_llm_response(prompt):
    """模拟LLM响应（演示用）"""
    # 简单的响应模拟，实际应用中应调用真实的LLM API
    
    # 检查提示中是否包含相关记忆
    has_software_engineer = "软件工程师" in prompt
    has_reading_traveling = "阅读" in prompt and "旅行" in prompt
    has_shanghai = "上海" in prompt
    
    # 根据当前问题和相关记忆生成响应
    if "天气" in prompt.split("用户:")[-1]:
        return "今天天气晴朗，温度适宜。"
    elif "名字" in prompt.split("用户:")[-1]:
        if "张三" in prompt:
            return "是的，您是张三。您之前告诉过我您的名字。"
        else:
            return "您还没有告诉我您的名字。"
    elif "爱好" in prompt.split("用户:")[-1]:
        if has_reading_traveling:
            return "根据您之前提到的，您喜欢阅读和旅行。"
        else:
            return "您还没有告诉我您的爱好。"
    elif "职业" in prompt.split("用户:")[-1]:
        if has_software_engineer:
            return "是的，我记得您是一名软件工程师，您在我们对话开始时告诉过我。"
        else:
            return "您还没有告诉我您的职业。"
    elif "旅行" in prompt.split("用户:")[-1] and "哪里" in prompt.split("用户:")[-1]:
        if has_shanghai:
            return "您告诉我您最近去了上海旅行。"
        else:
            return "您还没有告诉我您去哪里旅行了。"
    else:
        return "我理解您的问题。请告诉我更多信息，我可以更好地帮助您。"

class ScenarioMemory:
    """情景记忆存储（基于向量数据库的简化实现）"""
    
    def __init__(self):
        self.memories = []
        self.next_id = 1
    
    def add(self, memory_data):
        """添加新记忆"""
        memory_id = self.next_id
        self.next_id += 1
        
        memory = {
            'id': memory_id,
            'embedding': memory_data['embedding'],
            'content': memory_data['content'],
            'timestamp': memory_data['timestamp'],
            'importance': memory_data['importance'],
            'recall_count': memory_data['recall_count'],
            'last_access': memory_data['timestamp']
        }
        
        self.memories.append(memory)
        return memory_id
    
    def search(self, query_embedding=None, filter=None, limit=5):
        """搜索相关记忆"""
        results = []
        
        # 应用过滤器
        candidates = self.memories
        if filter:
            if 'user_id' in filter:
                # 在实际实现中，这里会过滤特定用户的记忆
                pass
                
            if 'last_access' in filter and '$lt' in filter['last_access']:
                threshold = filter['last_access']['$lt']
                candidates = [m for m in candidates if m['last_access'] < threshold]
                
            if 'importance' in filter and '$lt' in filter['importance']:
                threshold = filter['importance']['$lt']
                candidates = [m for m in candidates if m['importance'] < threshold]
        
        # 如果提供了查询嵌入，计算相似度
        if query_embedding is not None:
            for memory in candidates:
                similarity = cosine_similarity(
                    [query_embedding], 
                    [memory['embedding']]
                )[0][0]
                results.append((memory, similarity))
            
            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            results = [memory for memory, _ in results[:limit]]
        else:
            # 否则按时间排序
            candidates.sort(key=lambda x: x['recall_count'])
            results = candidates[:limit]
        
        # 更新访问时间
        for memory in results:
            memory['last_access'] = time.time()
            memory['recall_count'] += 1
            
        return results
    
    def update(self, memory_id, updates):
        """更新记忆"""
        for memory in self.memories:
            if memory['id'] == memory_id:
                for key, value in updates.items():
                    memory[key] = value
                return True
        return False

class KnowledgeGraph:
    """知识图谱存储（简化实现）"""
    
    def __init__(self):
        self.nodes = {}  # 实体节点
        self.relations = []  # 关系
    
    def add_or_update_node(self, entity):
        """添加或更新实体节点"""
        if entity['id'] not in self.nodes:
            self.nodes[entity['id']] = entity
        else:
            self.nodes[entity['id']].update(entity)
    
    def add_relation(self, source_id, relation_type, target_id, properties=None):
        """添加关系"""
        relation = {
            'source': source_id,
            'type': relation_type,
            'target': target_id,
            'properties': properties or {}
        }
        self.relations.append(relation)
    
    def find_related(self, entity_id, max_depth=1):
        """查找相关实体（简化实现）"""
        if max_depth <= 0 or entity_id not in self.nodes:
            return []
        
        related = []
        
        # 查找直接关联的实体
        for relation in self.relations:
            if relation['source'] == entity_id:
                target_id = relation['target']
                if target_id in self.nodes:
                    related.append({
                        'node': self.nodes[target_id],
                        'relation': relation
                    })
            
            elif relation['target'] == entity_id:
                source_id = relation['source']
                if source_id in self.nodes:
                    related.append({
                        'node': self.nodes[source_id],
                        'relation': relation
                    })
        
        # 递归查找更深层次的关联（简化实现）
        if max_depth > 1:
            for item in related.copy():
                node_id = item['node']['id']
                deeper_related = self.find_related(node_id, max_depth - 1)
                related.extend(deeper_related)
        
        return related

class WorkingMemory:
    """工作记忆（短期上下文缓存）"""
    
    def __init__(self, max_turns=10):
        self.conversations = {}
        self.max_turns = max_turns
    
    def get_conversation(self, conversation_id):
        """获取对话历史"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]
    
    def add_turn(self, conversation_id, user_input, assistant_response):
        """添加对话轮次"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        self.conversations[conversation_id].append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': time.time()
        })
        
        # 保持对话历史在最大轮次以内
        if len(self.conversations[conversation_id]) > self.max_turns:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_turns:]
    
    def get_context_string(self, conversation_id):
        """获取格式化的上下文字符串"""
        if conversation_id not in self.conversations:
            return ""
            
        context = ""
        for turn in self.conversations[conversation_id]:
            context += f"用户: {turn['user']}\n"
            context += f"助手: {turn['assistant']}\n"
        
        return context

class MemoryController:
    """记忆控制器（核心组件）"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.scenario_memory = ScenarioMemory()
        self.knowledge_graph = KnowledgeGraph()
        self.working_memory = WorkingMemory()
        
        # 记忆参数
        self.DECAY_RATE = 0.1  # 记忆衰减率
        self.RECALL_THRESHOLD = 0.3  # 记忆回忆阈值
        self.MIN_RECALL_PROBABILITY = 0.05  # 最小回忆概率
        self.LOW_IMPORTANCE_THRESHOLD = 0.3  # 低重要性阈值
        self.FORGETTING_THRESHOLD = 7 * 24 * 3600  # 遗忘阈值（7天）
    
    def extract_entities(self, text):
        """从文本中提取实体（简化实现）"""
        # 实际应用中应使用NER模型或知识图谱工具
        entities = []
        
        # 简单的关键词匹配
        keywords = {
            "职业": ["工程师", "医生", "教师", "学生", "研究员"],
            "爱好": ["阅读", "旅行", "音乐", "电影", "运动"],
            "地点": ["北京", "上海", "广州", "深圳"]
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in text:
                    entity_id = f"{category}:{word}"
                    entities.append({
                        'id': entity_id,
                        'type': category,
                        'name': word,
                        'source': 'extraction'
                    })
        
        return entities
    
    def extract_cues(self, text):
        """提取记忆线索（简化实现）"""
        # 实际应用中应使用更复杂的语义分析
        cues = []
        
        # 简单的关键词匹配
        keywords = {
            "职业": ["工作", "职业", "工程师", "医生", "教师"],
            "爱好": ["喜欢", "爱好", "兴趣", "阅读", "旅行"],
            "个人信息": ["名字", "年龄", "生日", "家庭"]
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in text:
                    cues.append(category)
                    break
        
        return cues
    
    def summarize_interaction(self, interaction):
        """总结交互内容（简化实现）"""
        # 实际应用中应使用LLM进行更智能的总结
        user_text = interaction.get('user', '')
        assistant_text = interaction.get('assistant', '')
        
        # 简单拼接
        summary = f"用户询问: {user_text[:50]}... 助手回复: {assistant_text[:50]}..."
        if len(user_text) <= 50:
            summary = f"用户询问: {user_text} 助手回复: {assistant_text[:50]}..."
            
        return summary
    
    def evaluate_importance(self, user_input, response):
        """评估交互重要性（简化实现）"""
        # 实际应用中应使用更复杂的重要性评估
        importance = 0.5  # 默认中等重要性
        
        # 包含个人信息的交互更重要
        personal_keywords = ["我是", "我的", "我喜欢", "我想", "我需要", "我工作"]
        for keyword in personal_keywords:
            if keyword in user_input:
                importance += 0.1
        
        # 问题和回答较长的交互更重要
        if len(user_input) > 50:
            importance += 0.1
        if len(response) > 100:
            importance += 0.1
            
        return min(1.0, importance)  # 确保重要性在0-1之间
    
    def calculate_recall_probability(self, relevance, time_elapsed, recall_count):
        """计算记忆回忆概率"""
        # 基于艾宾浩斯遗忘曲线的修改版本
        time_factor = time_elapsed / (24 * 3600)  # 转换为天数
        base_recall = relevance * math.exp(-self.DECAY_RATE * time_factor / (recall_count + 1))
        
        # 确保记忆永远不会完全消失（无遗忘原则）
        recall_prob = max(self.MIN_RECALL_PROBABILITY, base_recall)
        
        return recall_prob
    
    def calculate_forget_probability(self, importance, time_elapsed, recall_count):
        """计算记忆遗忘概率"""
        # 重要性越高，遗忘概率越低
        importance_factor = 1 - importance
        
        # 时间越长，遗忘概率越高
        time_factor = min(1.0, time_elapsed / (30 * 24 * 3600))  # 最多考虑30天
        
        # 回忆次数越多，遗忘概率越低
        recall_factor = math.exp(-0.5 * recall_count)
        
        forget_prob = importance_factor * time_factor * recall_factor
        return forget_prob
    
    def compress_memory(self, content, compression_ratio=0.5):
        """压缩记忆内容（简化实现）"""
        # 实际应用中应使用LLM进行智能压缩
        words = content.split()
        compressed_length = max(10, int(len(words) * compression_ratio))
        compressed = ' '.join(words[:compressed_length]) + "..."
        return compressed
    
    def consolidate_memory(self, interaction, importance):
        """将交互巩固到长期记忆"""
        # 1. 提取关键信息
        summary = self.summarize_interaction(interaction)
        entities = self.extract_entities(interaction['user'] + " " + interaction['assistant'])
        
        # 2. 更新情景记忆
        embedding = embed_text(summary)
        memory_id = self.scenario_memory.add({
            'embedding': embedding,
            'content': summary,
            'timestamp': time.time(),
            'importance': importance,
            'recall_count': 1
        })
        
        # 3. 更新知识图谱
        for entity in entities:
            self.knowledge_graph.add_or_update_node(entity)
            self.knowledge_graph.add_relation(
                self.user_id, 
                'mentioned', 
                entity['id'], 
                {'memory_id': memory_id, 'timestamp': time.time()}
            )
        
        return memory_id
    
    def retrieve_memories(self, current_context, max_memories=3):
        """基于当前上下文检索相关记忆"""
        # 1. 提取检索线索
        retrieval_cues = self.extract_cues(current_context)
        
        # 2. 向量检索（情景记忆）
        context_embedding = embed_text(current_context)
        scenario_results = self.scenario_memory.search(
            query_embedding=context_embedding,
            filter={"user_id": self.user_id},
            limit=max_memories*2  # 检索更多候选项
        )
        
        # 3. 图检索（知识图谱）- 简化实现
        graph_results = []
        for cue in retrieval_cues:
            entity_id = f"{cue}"
            related = self.knowledge_graph.find_related(entity_id, max_depth=1)
            for item in related:
                if 'memory_id' in item['relation']['properties']:
                    memory_id = item['relation']['properties']['memory_id']
                    # 在实际实现中，这里应该从情景记忆中获取对应的记忆
                    # 这里简化处理
                    for memory in scenario_results:
                        if memory['id'] == memory_id:
                            graph_results.append(memory)
        
        # 4. 融合结果并计算记忆激活概率
        combined_results = []
        current_time = time.time()
        
        for memory in scenario_results + graph_results:
            # 避免重复
            if any(m['id'] == memory['id'] for m, _ in combined_results):
                continue
                
            relevance = cosine_similarity(
                [context_embedding], 
                [memory['embedding']]
            )[0][0]
            
            recall_prob = self.calculate_recall_probability(
                relevance=relevance,
                time_elapsed=current_time - memory['timestamp'],
                recall_count=memory['recall_count']
            )
            
            if recall_prob > self.RECALL_THRESHOLD:
                combined_results.append((memory, recall_prob))
        
        # 5. 排序并返回最相关记忆
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in combined_results[:max_memories]]
    
    def strategic_forgetting(self):
        """策略性遗忘机制"""
        print("\n执行策略性遗忘...")
        
        # 1. 获取低重要性且长时间未访问的记忆
        current_time = time.time()
        candidates = self.scenario_memory.search(
            filter={
                "last_access": {"$lt": current_time - self.FORGETTING_THRESHOLD},
                "importance": {"$lt": self.LOW_IMPORTANCE_THRESHOLD}
            },
            limit=5
        )
        
        # 2. 计算遗忘概率并应用遗忘
        for memory in candidates:
            forget_prob = self.calculate_forget_probability(
                importance=memory['importance'],
                time_elapsed=current_time - memory['timestamp'],
                recall_count=memory['recall_count']
            )
            
            print(f"记忆: {memory['content']}")
            print(f"重要性: {memory['importance']:.2f}, 上次访问: {(current_time - memory['last_access']) / 3600:.1f}小时前")
            print(f"遗忘概率: {forget_prob:.2f}")
            
            # 3. 应用遗忘（注意：不是完全删除，而是降低精度）
            if random.random() < forget_prob:
                print("应用遗忘 -> 压缩记忆")
                compressed_content = self.compress_memory(memory['content'])
                self.scenario_memory.update(memory['id'], {
                    'content': compressed_content, 
                    'precision_level': 'reduced'
                })
                print(f"压缩后: {compressed_content}")
            else:
                print("保留记忆")
            print("-" * 40)
    
    def get_working_memory(self, conversation_id):
        """获取工作记忆（当前对话上下文）"""
        return self.working_memory.get_context_string(conversation_id)
    
    def update_working_memory(self, conversation_id, user_input, response):
        """更新工作记忆"""
        self.working_memory.add_turn(conversation_id, user_input, response)
    
    def build_enhanced_prompt(self, user_input, current_context, relevant_memories):
        """构建增强提示"""
        prompt = "系统: 你是一个具有记忆能力的AI助手。请基于以下信息回答用户的问题。\n\n"
        
        # 添加相关记忆
        if relevant_memories:
            prompt += "相关记忆:\n"
            for memory in relevant_memories:
                prompt += f"- {memory['content']}\n"
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

class MemoryXAdapter:
    """适配任意LLM的记忆增强接口"""
    
    def __init__(self, llm_provider, user_id):
        self.llm = llm_provider
        self.user_id = user_id
        self.memory_controller = MemoryController(user_id)
        self.conversation_id = "demo_conversation"  # 演示用固定会话ID
        
    def generate_response(self, user_input):
        """生成增强记忆的响应"""
        print("\n处理用户输入:", user_input)
        
        # 1. 获取当前对话上下文
        current_context = self.memory_controller.get_working_memory(self.conversation_id)
        print("\n当前对话上下文:", current_context if current_context else "[无上下文]")
        
        # 2. 检索相关长期记忆
        relevant_memories = self.memory_controller.retrieve_memories(
            current_context + user_input
        )
        
        if relevant_memories:
            print("\n检索到的相关记忆:")
            for i, memory in enumerate(relevant_memories):
                print(f"{i+1}. {memory['content']} (重要性: {memory['importance']:.2f}, 回忆次数: {memory['recall_count']})")
        else:
            print("\n未检索到相关记忆")
        
        # 3. 构建增强提示
        enhanced_prompt = self.memory_controller.build_enhanced_prompt(
            user_input=user_input,
            current_context=current_context,
            relevant_memories=relevant_memories
        )
        
        print("\n增强提示:")
        print("-" * 60)
        print(enhanced_prompt)
        print("-" * 60)
        
        # 4. 调用LLM生成响应
        response = self.llm(enhanced_prompt)
        print("\nLLM响应:", response)
        
        # 5. 更新记忆
        importance = self.memory_controller.evaluate_importance(user_input, response)
        print(f"\n交互重要性评分: {importance:.2f}")
        
        self.memory_controller.update_working_memory(self.conversation_id, user_input, response)
        self.memory_controller.consolidate_memory(
            interaction={'user': user_input, 'assistant': response},
            importance=importance
        )
        
        # 6. 定期执行记忆管理
        if random.random() < 0.3:  # 30%概率执行记忆管理
            self.memory_controller.strategic_forgetting()
            
        return response

# 演示函数
def run_demo():
    """运行MemoryX演示"""
    print("=" * 80)
    print("MemoryX: 通用LLM记忆增强模块演示")
    print("=" * 80)
    
    # 初始化适配器
    adapter = MemoryXAdapter(simulate_llm_response, "demo_user")
    
    # 模拟对话
    conversations = [
        "你好，我是张三，我是一名软件工程师。",
        "我喜欢阅读和旅行。",
        "今天天气怎么样？",
        "你还记得我的名字吗？",
        "我的职业是什么？",
        "我有什么爱好？",
        "我住在北京。",
        "我最近去了上海旅行。",
        "你还记得我去哪里旅行了吗？",
        "我的职业是什么？"  # 重复问题，测试记忆强化
    ]
    
    for i, user_input in enumerate(conversations):
        print("\n" + "=" * 80)
        print(f"对话轮次 {i+1}/{len(conversations)}")
        print("=" * 80)
        
        response = adapter.generate_response(user_input)
        
        print("\n最终响应:", response)
        
        # 模拟时间流逝
        if i < len(conversations) - 1:
            time_skip = random.randint(1, 5)
            print(f"\n[时间流逝: {time_skip} 小时后...]")
            # 修改记忆时间戳以模拟时间流逝
            for memory in adapter.memory_controller.scenario_memory.memories:
                memory['timestamp'] -= time_skip * 3600
                memory['last_access'] -= time_skip * 3600

if __name__ == "__main__":
    run_demo()