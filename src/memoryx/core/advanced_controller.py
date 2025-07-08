"""
MemoryX 高级模块: 优化搜索算法与增强记忆压缩
==============================================

本模块扩展了基础MemoryX功能，重点优化以下方面:
1. 高级记忆检索算法 - 混合检索与重排序
2. 智能记忆压缩 - 基于重要性的层次化压缩
3. 记忆重要性评估 - 多维度评分系统
4. 层次化记忆架构 - 短期、中期和长期记忆管理
"""

import os
import math
import time
import json
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ===============================
# 高级嵌入与相似度计算
# ===============================

class AdvancedEmbedding:
    """高级嵌入模型，支持多种嵌入策略"""
    
    def __init__(self, model_name="simulated"):
        self.model_name = model_name
        self.embedding_dim = 384  # 实际应用中应根据选择的模型调整
        
    def embed_text(self, text, strategy="default"):
        """生成文本嵌入向量
        
        策略:
        - default: 标准嵌入
        - semantic: 优化语义理解的嵌入
        - factual: 优化事实提取的嵌入
        """
        # 实际应用中应使用真实的嵌入模型，如OpenAI的嵌入API
        if self.model_name == "simulated":
            # 模拟不同嵌入策略
            random.seed(hash(text) % 10000)
            base_embedding = np.array([random.random() for _ in range(self.embedding_dim)])
            
            if strategy == "semantic":
                # 模拟语义优化的嵌入
                semantic_factor = np.array([0.2 * math.sin(i) for i in range(self.embedding_dim)])
                return base_embedding + semantic_factor
            elif strategy == "factual":
                # 模拟事实优化的嵌入
                factual_factor = np.array([0.2 * math.cos(i) for i in range(self.embedding_dim)])
                return base_embedding + factual_factor
            else:
                return base_embedding
        else:
            # 实际应用中的代码，调用真实的嵌入模型
            raise NotImplementedError("仅支持模拟嵌入模型")
    
    def compute_similarity(self, embedding1, embedding2, method="cosine"):
        """计算两个嵌入向量之间的相似度
        
        方法:
        - cosine: 余弦相似度
        - dot: 点积
        - weighted: 加权余弦相似度
        """
        if method == "cosine":
            return cosine_similarity([embedding1], [embedding2])[0][0]
        elif method == "dot":
            return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        elif method == "weighted":
            # 加权余弦相似度，可以根据特定领域知识调整权重
            weights = np.ones(self.embedding_dim)
            # 示例：增加前100维的权重
            weights[:100] = 1.5
            weighted_emb1 = embedding1 * weights
            weighted_emb2 = embedding2 * weights
            return cosine_similarity([weighted_emb1], [weighted_emb2])[0][0]
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")

# ===============================
# 高级记忆存储与检索
# ===============================

class HierarchicalMemory:
    """层次化记忆存储，包含短期、中期和长期记忆"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        # 短期记忆 - 最近的交互，无需压缩
        self.short_term_memory = []
        self.short_term_capacity = 20
        
        # 中期记忆 - 部分压缩的重要记忆
        self.medium_term_memory = []
        self.medium_term_capacity = 100
        
        # 长期记忆 - 高度压缩的核心记忆
        self.long_term_memory = []
        
        # 记忆索引 - 用于快速检索
        self.memory_index = defaultdict(list)  # 关键词 -> 记忆ID列表
        
        # 记忆聚类 - 用于相似记忆的组织
        self.clusters = None
        self.cluster_centers = None
        self.memory_to_cluster = {}
        
        self.next_id = 1
    
    def add_memory(self, content, metadata=None, importance=0.5):
        """添加新记忆"""
        memory_id = self.next_id
        self.next_id += 1
        
        # 生成嵌入
        embedding = self.embedding_model.embed_text(content)
        
        # 创建记忆对象
        memory = {
            'id': memory_id,
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {},
            'importance': importance,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'access_count': 1,
            'compression_level': 0,  # 0=无压缩, 1=轻度压缩, 2=中度压缩, 3=高度压缩
            'original_content': content  # 保留原始内容用于评估压缩质量
        }
        
        # 添加到短期记忆
        self.short_term_memory.append(memory)
        
        # 如果短期记忆超出容量，触发记忆整合
        if len(self.short_term_memory) > self.short_term_capacity:
            self._consolidate_memories()
        
        # 更新索引
        self._index_memory(memory)
        
        return memory_id
    
    def _index_memory(self, memory):
        """为记忆创建索引"""
        # 提取关键词（实际应用中应使用更复杂的关键词提取算法）
        keywords = self._extract_keywords(memory['content'])
        
        # 更新索引
        for keyword in keywords:
            self.memory_index[keyword].append(memory['id'])
    
    def _extract_keywords(self, text, max_keywords=5):
        """从文本中提取关键词（简化实现）"""
        # 实际应用中应使用NLP技术如TF-IDF或关键词提取模型
        words = text.lower().split()
        # 移除常见停用词
        stopwords = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
        
        # 选择最频繁的词作为关键词
        word_freq = defaultdict(int)
        for word in filtered_words:
            word_freq[word] += 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _consolidate_memories(self):
        """记忆整合过程 - 将短期记忆转移到中期或长期记忆"""
        print("\n执行记忆整合...")
        
        # 1. 评估短期记忆的重要性
        for memory in self.short_term_memory:
            # 更新重要性评分
            memory['importance'] = self._evaluate_importance(memory)
        
        # 2. 按重要性排序
        self.short_term_memory.sort(key=lambda x: x['importance'], reverse=True)
        
        # 3. 将重要的记忆转移到中期记忆（轻度压缩）
        important_memories = self.short_term_memory[:len(self.short_term_memory)//2]
        for memory in important_memories:
            # 轻度压缩
            memory['content'] = self._compress_memory(memory['content'], level=1)
            memory['compression_level'] = 1
            self.medium_term_memory.append(memory)
            print(f"记忆 {memory['id']} 转移到中期记忆，重要性: {memory['importance']:.2f}")
        
        # 4. 将不太重要的记忆丢弃或高度压缩后存入长期记忆
        less_important_memories = self.short_term_memory[len(self.short_term_memory)//2:]
        for memory in less_important_memories:
            if memory['importance'] > 0.3:  # 仍有一定重要性
                # 中度压缩
                memory['content'] = self._compress_memory(memory['content'], level=2)
                memory['compression_level'] = 2
                self.long_term_memory.append(memory)
                print(f"记忆 {memory['id']} 转移到长期记忆，重要性: {memory['importance']:.2f}")
            else:
                print(f"记忆 {memory['id']} 被丢弃，重要性过低: {memory['importance']:.2f}")
        
        # 5. 清空短期记忆
        self.short_term_memory = []
        
        # 6. 如果中期记忆超出容量，压缩并转移到长期记忆
        if len(self.medium_term_memory) > self.medium_term_capacity:
            self._compress_medium_term_memory()
        
        # 7. 定期对长期记忆进行聚类，便于组织和检索
        if len(self.long_term_memory) > 50 and (len(self.long_term_memory) % 20 == 0):
            self._cluster_long_term_memories()
    
    def _compress_medium_term_memory(self):
        """压缩中期记忆并转移到长期记忆"""
        print("\n压缩中期记忆...")
        
        # 按最后访问时间和重要性排序
        self.medium_term_memory.sort(key=lambda x: (time.time() - x['last_accessed']) / x['importance'])
        
        # 将最老且最不重要的记忆转移到长期记忆
        memories_to_move = self.medium_term_memory[self.medium_term_capacity//2:]
        for memory in memories_to_move:
            # 进一步压缩
            memory['content'] = self._compress_memory(memory['content'], level=3)
            memory['compression_level'] = 3
            self.long_term_memory.append(memory)
            print(f"中期记忆 {memory['id']} 压缩并转移到长期记忆")
        
        # 更新中期记忆
        self.medium_term_memory = self.medium_term_memory[:self.medium_term_capacity//2]
    
    def _cluster_long_term_memories(self, n_clusters=10):
        """对长期记忆进行聚类"""
        if len(self.long_term_memory) < n_clusters:
            return
            
        print("\n对长期记忆进行聚类...")
        
        # 提取所有嵌入
        embeddings = np.array([memory['embedding'] for memory in self.long_term_memory])
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # 保存聚类结果
        self.clusters = clusters
        self.cluster_centers = kmeans.cluster_centers_
        
        # 将记忆映射到聚类
        for i, memory in enumerate(self.long_term_memory):
            self.memory_to_cluster[memory['id']] = clusters[i]
        
        # 打印聚类统计
        cluster_counts = defaultdict(int)
        for cluster_id in clusters:
            cluster_counts[cluster_id] += 1
        
        print(f"长期记忆聚类完成，共 {n_clusters} 个聚类:")
        for cluster_id, count in cluster_counts.items():
            print(f"  聚类 {cluster_id}: {count} 个记忆")
    
    def _evaluate_importance(self, memory):
        """评估记忆的重要性（多维度评分）"""
        base_importance = memory['importance']
        
        # 1. 时间衰减因子 - 较新的记忆更重要
        time_factor = math.exp(-0.01 * (time.time() - memory['created_at']) / 3600)  # 每小时衰减约1%
        
        # 2. 访问频率因子 - 经常访问的记忆更重要
        access_factor = min(1.0, 0.1 * math.log(1 + memory['access_count']))
        
        # 3. 情感强度因子 - 情感强烈的记忆更重要（实际应用中应使用情感分析）
        emotion_factor = 0.0
        if 'emotion' in memory['metadata']:
            emotion_factor = min(0.3, abs(memory['metadata']['emotion']) / 5.0)
        
        # 4. 关联度因子 - 与其他记忆高度关联的记忆更重要
        relation_factor = 0.0
        if 'relation_count' in memory['metadata']:
            relation_factor = min(0.2, 0.05 * memory['metadata']['relation_count'])
        
        # 综合评分，确保在0-1范围内
        importance = min(1.0, base_importance + time_factor * 0.2 + access_factor + emotion_factor + relation_factor)
        
        return importance
    
    def _compress_memory(self, content, level=1):
        """多级记忆压缩
        
        级别:
        1 = 轻度压缩 (保留约75%内容)
        2 = 中度压缩 (保留约50%内容)
        3 = 高度压缩 (保留约25%内容)
        """
        # 实际应用中应使用LLM或专门的摘要模型进行智能压缩
        
        # 简化实现：根据压缩级别保留部分内容
        words = content.split()
        if level == 1:
            # 轻度压缩：保留前75%的内容
            keep_ratio = 0.75
        elif level == 2:
            # 中度压缩：保留前50%的内容
            keep_ratio = 0.5
        else:
            # 高度压缩：保留前25%的内容
            keep_ratio = 0.25
        
        keep_words = int(len(words) * keep_ratio)
        if keep_words < 3:
            keep_words = min(3, len(words))
            
        compressed = ' '.join(words[:keep_words])
        if keep_words < len(words):
            compressed += "..."
            
        return compressed
    
    def search(self, query, search_type="hybrid", top_k=5):
        """高级记忆搜索
        
        搜索类型:
        - semantic: 语义搜索（基于嵌入相似度）
        - keyword: 关键词搜索（基于索引）
        - hybrid: 混合搜索（结合语义和关键词）
        - temporal: 时间感知搜索（考虑时间因素）
        """
        print(f"\n执行{search_type}搜索: {query}")
        
        if search_type == "semantic":
            return self._semantic_search(query, top_k)
        elif search_type == "keyword":
            return self._keyword_search(query, top_k)
        elif search_type == "temporal":
            return self._temporal_search(query, top_k)
        else:  # hybrid (默认)
            return self._hybrid_search(query, top_k)
    
    def _semantic_search(self, query, top_k=5):
        """语义搜索 - 基于嵌入相似度"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_text(query, strategy="semantic")
        
        # 合并所有记忆
        all_memories = self.short_term_memory + self.medium_term_memory + self.long_term_memory
        
        # 计算相似度并排序
        results = []
        for memory in all_memories:
            similarity = self.embedding_model.compute_similarity(
                query_embedding, memory['embedding'], method="cosine"
            )
            results.append((memory, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问统计
        for memory, _ in results[:top_k]:
            memory['last_accessed'] = time.time()
            memory['access_count'] += 1
        
        return [memory for memory, _ in results[:top_k]]
    
    def _keyword_search(self, query, top_k=5):
        """关键词搜索 - 基于索引"""
        # 提取查询关键词
        query_keywords = self._extract_keywords(query)
        
        # 查找匹配的记忆ID
        matching_ids = set()
        for keyword in query_keywords:
            matching_ids.update(self.memory_index.get(keyword, []))
        
        # 获取匹配的记忆
        all_memories = self.short_term_memory + self.medium_term_memory + self.long_term_memory
        memory_dict = {memory['id']: memory for memory in all_memories}
        
        matching_memories = [memory_dict[id] for id in matching_ids if id in memory_dict]
        
        # 按关键词匹配数量排序
        results = []
        for memory in matching_memories:
            memory_keywords = self._extract_keywords(memory['content'])
            match_count = sum(1 for k in query_keywords if k in memory_keywords)
            results.append((memory, match_count))
        
        # 按匹配数量排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问统计
        for memory, _ in results[:top_k]:
            memory['last_accessed'] = time.time()
            memory['access_count'] += 1
        
        return [memory for memory, _ in results[:top_k]]
    
    def _temporal_search(self, query, top_k=5, recency_weight=0.3):
        """时间感知搜索 - 考虑相似度和时间因素"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_text(query)
        
        # 合并所有记忆
        all_memories = self.short_term_memory + self.medium_term_memory + self.long_term_memory
        
        # 计算相似度和时间因子
        current_time = time.time()
        results = []
        for memory in all_memories:
            # 语义相似度
            similarity = self.embedding_model.compute_similarity(
                query_embedding, memory['embedding']
            )
            
            # 时间因子 - 较新的记忆得分更高
            time_diff = (current_time - memory['created_at']) / (24 * 3600)  # 天数
            recency_factor = math.exp(-0.1 * time_diff)  # 每10天衰减约63%
            
            # 综合得分
            combined_score = similarity * (1 - recency_weight) + recency_factor * recency_weight
            
            results.append((memory, combined_score))
        
        # 按综合得分排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 更新访问统计
        for memory, _ in results[:top_k]:
            memory['last_accessed'] = time.time()
            memory['access_count'] += 1
        
        return [memory for memory, _ in results[:top_k]]
    
    def _hybrid_search(self, query, top_k=5):
        """混合搜索 - 结合语义、关键词和聚类"""
        # 1. 语义搜索
        semantic_results = self._semantic_search(query, top_k=top_k*2)
        semantic_ids = {memory['id'] for memory in semantic_results}
        
        # 2. 关键词搜索
        keyword_results = self._keyword_search(query, top_k=top_k*2)
        keyword_ids = {memory['id'] for memory in keyword_results}
        
        # 3. 合并结果并去重
        combined_results = []
        
        # 首先添加同时出现在两种搜索中的结果
        for memory in semantic_results:
            if memory['id'] in keyword_ids:
                combined_results.append(memory)
        
        # 然后添加仅在语义搜索中的结果
        for memory in semantic_results:
            if memory['id'] not in keyword_ids and memory not in combined_results:
                combined_results.append(memory)
        
        # 最后添加仅在关键词搜索中的结果
        for memory in keyword_results:
            if memory['id'] not in semantic_ids and memory not in combined_results:
                combined_results.append(memory)
        
        # 4. 如果结果不足，尝试从相同聚类中添加相关记忆
        if len(combined_results) < top_k and self.clusters is not None:
            # 获取已有结果的聚类
            result_clusters = set()
            for memory in combined_results:
                if memory['id'] in self.memory_to_cluster:
                    result_clusters.add(self.memory_to_cluster[memory['id']])
            
            # 从相同聚类中添加记忆
            for memory in self.long_term_memory:
                if (memory['id'] in self.memory_to_cluster and 
                    self.memory_to_cluster[memory['id']] in result_clusters and
                    memory not in combined_results):
                    combined_results.append(memory)
                    if len(combined_results) >= top_k:
                        break
        
        # 5. 更新访问统计
        for memory in combined_results[:top_k]:
            memory['last_accessed'] = time.time()
            memory['access_count'] += 1
        
        return combined_results[:top_k]

# ===============================
# 高级记忆控制器
# ===============================

class AdvancedMemoryController:
    """高级记忆控制器，管理记忆的存储、检索和整合"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.embedding_model = AdvancedEmbedding()
        self.memory_store = HierarchicalMemory(self.embedding_model)
        
        # 工作记忆（对话上下文）
        self.working_memory = {}
        self.max_turns = 10
        
        # 记忆检索配置
        self.recall_threshold = 0.3
        self.max_memories_per_query = 5
    
    def add_interaction(self, user_input, assistant_response, conversation_id="default"):
        """添加新的交互到记忆"""
        # 1. 更新工作记忆
        self._update_working_memory(conversation_id, user_input, assistant_response)
        
        # 2. 评估交互重要性
        importance = self._evaluate_interaction_importance(user_input, assistant_response)
        
        # 3. 提取元数据
        metadata = self._extract_metadata(user_input, assistant_response)
        
        # 4. 创建记忆内容
        memory_content = f"用户: {user_input}\n助手: {assistant_response}"
        
        # 5. 添加到记忆存储
        memory_id = self.memory_store.add_memory(
            content=memory_content,
            metadata=metadata,
            importance=importance
        )
        
        return memory_id, importance
    
    def retrieve_memories(self, query, context=None, search_type="hybrid"):
        """检索相关记忆"""
        # 1. 增强查询（结合上下文）
        enhanced_query = query
        if context:
            enhanced_query = f"{context}\n{query}"
        
        # 2. 执行记忆搜索
        relevant_memories = self.memory_store.search(
            query=enhanced_query,
            search_type=search_type,
            top_k=self.max_memories_per_query
        )
        
        return relevant_memories
    
    def get_working_memory(self, conversation_id="default"):
        """获取工作记忆（对话上下文）"""
        if conversation_id not in self.working_memory:
            return ""
            
        context = ""
        for turn in self.working_memory[conversation_id]:
            context += f"用户: {turn['user']}\n"
            context += f"助手: {turn['assistant']}\n"
        
        return context
    
    def _update_working_memory(self, conversation_id, user_input, assistant_response):
        """更新工作记忆"""
        if conversation_id not in self.working_memory:
            self.working_memory[conversation_id] = []
            
        self.working_memory[conversation_id].append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': time.time()
        })
        
        # 保持对话历史在最大轮次以内
        if len(self.working_memory[conversation_id]) > self.max_turns:
            self.working_memory[conversation_id] = self.working_memory[conversation_id][-self.max_turns:]
    
    def _evaluate_interaction_importance(self, user_input, assistant_response):
        """评估交互的重要性（多维度评分）"""
        importance = 0.5  # 默认中等重要性
        
        # 1. 长度因子 - 较长的交互可能更重要
        length_factor = min(0.2, (len(user_input) + len(assistant_response)) / 1000)
        
        # 2. 个人信息因子 - 包含个人信息的交互更重要
        personal_keywords = ["我是", "我的", "我喜欢", "我想", "我需要", "我工作", "我住在"]
        personal_factor = 0
        for keyword in personal_keywords:
            if keyword in user_input:
                personal_factor += 0.05
        personal_factor = min(0.3, personal_factor)
        
        # 3. 问答质量因子 - 有实质性回答的交互更重要
        qa_factor = 0
        if len(assistant_response) > 100:
            qa_factor += 0.1
        if "?" in user_input and len(assistant_response) > 50:
            qa_factor += 0.1
        qa_factor = min(0.2, qa_factor)
        
        # 4. 情感因子 - 情感强烈的交互更重要（实际应用中应使用情感分析）
        emotion_keywords = ["喜欢", "爱", "讨厌", "恨", "开心", "难过", "生气", "惊讶", "害怕"]
        emotion_factor = 0
        for keyword in emotion_keywords:
            if keyword in user_input:
                emotion_factor += 0.05
        emotion_factor = min(0.2, emotion_factor)
        
        # 综合评分，确保在0-1范围内
        importance += length_factor + personal_factor + qa_factor + emotion_factor
        importance = max(0.1, min(1.0, importance))
        
        return importance
    
    def _extract_metadata(self, user_input, assistant_response):
        """从交互中提取元数据"""
        metadata = {
            'timestamp': time.time(),
            'length': len(user_input) + len(assistant_response),
            'has_question': '?' in user_input,
            'keywords': self.memory_store._extract_keywords(user_input + " " + assistant_response)
        }
        
        # 提取实体（简化实现）
        entities = self._extract_entities(user_input + " " + assistant_response)
        if entities:
            metadata['entities'] = entities
        
        # 提取情感（简化实现）
        emotion = self._analyze_emotion(user_input)
        if emotion:
            metadata['emotion'] = emotion
        
        return metadata
    
    def _extract_entities(self, text):
        """从文本中提取实体（简化实现）"""
        entities = []
        
        # 简单的关键词匹配
        entity_patterns = {
            "人物": ["张三", "李四", "王五"],
            "地点": ["北京", "上海", "广州", "深圳"],
            "职业": ["工程师", "医生", "教师", "学生", "研究员"],
            "爱好": ["阅读", "旅行", "音乐", "电影", "运动"]
        }
        
        for category, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    entities.append({
                        'type': category,
                        'value': pattern,
                        'position': text.find(pattern)
                    })
        
        return entities
    
    def _analyze_emotion(self, text):
        """简单的情感分析（实际应用中应使用更复杂的情感分析模型）"""
        positive_words = ["喜欢", "爱", "开心", "高兴", "好", "棒", "优秀", "感谢"]
        negative_words = ["讨厌", "恨", "难过", "伤心", "不好", "糟糕", "失望", "抱歉"]
        
        # 计算情感得分（-1到1之间）
        score = 0
        for word in positive_words:
            if word in text:
                score += 0.2
        for word in negative_words:
            if word in text:
                score -= 0.2
        
        return max(-1.0, min(1.0, score))
    
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

# ===============================
# 高级记忆增强适配器
# ===============================

class AdvancedMemoryAdapter:
    """高级记忆增强适配器，集成LLM与记忆系统"""
    
    def __init__(self, llm_provider, user_id):
        self.llm = llm_provider
        self.user_id = user_id
        self.memory_controller = AdvancedMemoryController(user_id)
        self.conversation_id = "demo_conversation"  # 演示用固定会话ID
        
        # 记忆检索配置
        self.search_strategies = {
            "general": "hybrid",      # 一般问题
            "factual": "semantic",    # 事实性问题
            "temporal": "temporal",   # 时间相关问题
            "personal": "hybrid"      # 个人信息问题
        }
    
    def generate_response(self, user_input):
        """生成增强记忆的响应"""
        print("\n处理用户输入:", user_input)
        
        # 1. 确定查询类型
        query_type = self._determine_query_type(user_input)
        search_type = self.search_strategies.get(query_type, "hybrid")
        print(f"查询类型: {query_type}, 搜索策略: {search_type}")
        
        # 2. 获取当前对话上下文
        current_context = self.memory_controller.get_working_memory(self.conversation_id)
        print("\n当前对话上下文:", current_context if current_context else "[无上下文]")
        
        # 3. 检索相关长期记忆
        relevant_memories = self.memory_controller.retrieve_memories(
            query=user_input,
            context=current_context,
            search_type=search_type
        )
        
        if relevant_memories:
            print("\n检索到的相关记忆:")
            for i, memory in enumerate(relevant_memories):
                print(f"{i+1}. {memory['content']} (重要性: {memory['importance']:.2f}, 压缩级别: {memory['compression_level']})")
        else:
            print("\n未检索到相关记忆")
        
        # 4. 构建增强提示
        enhanced_prompt = self.memory_controller.build_enhanced_prompt(
            user_input=user_input,
            current_context=current_context,
            relevant_memories=relevant_memories
        )
        
        print("\n增强提示:")
        print("-" * 60)
        print(enhanced_prompt)
        print("-" * 60)
        
        # 5. 调用LLM生成响应
        response = self.llm(enhanced_prompt)
        print("\nLLM响应:", response)
        
        # 6. 更新记忆
        memory_id, importance = self.memory_controller.add_interaction(
            user_input=user_input,
            assistant_response=response,
            conversation_id=self.conversation_id
        )
        
        print(f"\n交互已存储为记忆 ID: {memory_id}, 重要性评分: {importance:.2f}")
        
        return response
    
    def _determine_query_type(self, query):
        """确定查询类型，以选择最佳搜索策略"""
        # 事实性问题
        if any(word in query.lower() for word in ["是什么", "为什么", "如何", "怎么样", "定义"]):
            return "factual"
        
        # 时间相关问题
        if any(word in query.lower() for word in ["什么时候", "多久", "几点", "日期", "最近", "上次"]):
            return "temporal"
        
        # 个人信息问题
        if any(word in query.lower() for word in ["我的", "我是", "我有", "记得我", "告诉过你"]):
            return "personal"
        
        # 默认为一般问题
        return "general"

# ===============================
# 模拟LLM响应函数
# ===============================

def simulate_llm_response(prompt):
    """模拟LLM响应（演示用）"""
    # 检查提示中是否包含相关记忆
    has_software_engineer = "软件工程师" in prompt
    has_reading_traveling = "阅读" in prompt and "旅行" in prompt
    has_shanghai = "上海" in prompt
    has_beijing = "北京" in prompt
    
    # 根据当前问题和相关记忆生成响应
    if "总结" in prompt.split("用户:")[-1] or "了解" in prompt.split("用户:")[-1]:
        # 生成个人信息总结
        summary = "根据我们的对话，我了解到：\n"
        if "张三" in prompt:
            summary += "1. 您的名字是张三\n"
        if has_software_engineer:
            summary += "2. 您是一名软件工程师\n"
        if has_reading_traveling:
            summary += "3. 您喜欢阅读和旅行\n"
        if has_beijing:
            summary += "4. 您住在北京\n"
        if has_shanghai:
            summary += "5. 您最近去了上海旅行\n"
        return summary
    elif "天气" in prompt.split("用户:")[-1]:
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
    elif "住" in prompt.split("用户:")[-1] or "家" in prompt.split("用户:")[-1]:
        if has_beijing:
            return "您之前提到您住在北京。"
        else:
            return "您还没有告诉我您住在哪里。"
    else:
        return "我理解您的问题。请告诉我更多信息，我可以更好地帮助您。"

# ===============================
# 演示函数
# ===============================

def run_advanced_demo():
    """运行MemoryX高级功能演示"""
    print("=" * 80)
    print("MemoryX高级模块: 优化搜索算法与增强记忆压缩")
    print("=" * 80)
    
    # 初始化适配器
    adapter = AdvancedMemoryAdapter(simulate_llm_response, "demo_user")
    
    # 模拟对话
    conversations = [
        "你好，我是张三，我是一名软件工程师。",
        "我喜欢阅读和旅行。",
        "我住在北京。",
        "今天天气怎么样？",
        "你还记得我的名字吗？",
        "我的职业是什么？",
        "我有什么爱好？",
        "我最近去了上海旅行。",
        "你还记得我去哪里旅行了吗？",
        "我住在哪里？",
        "总结一下你对我的了解。"
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
            # 这里不需要手动修改时间戳，因为高级记忆控制器会自动处理时间因素

if __name__ == "__main__":
    run_advanced_demo()