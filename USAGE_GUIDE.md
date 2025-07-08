# MemoryX 使用指南

本文档提供了 MemoryX 多模态记忆系统的详细使用说明，包括安装、基本功能和高级功能的使用方法。

## 目录

1. [安装](#安装)
2. [基本文本记忆](#基本文本记忆)
3. [图像记忆](#图像记忆)
4. [多模态记忆](#多模态记忆)
5. [增强图像理解](#增强图像理解)
6. [高级配置](#高级配置)
7. [常见问题解答](#常见问题解答)

## 安装

### 从 GitHub 安装

```bash
# 克隆仓库
git clone https://github.com/yuzengbaao/MemoryX.git
cd MemoryX

# 安装依赖
pip install -e .
```

### 从 PyPI 安装 (即将推出)

```bash
pip install memoryx
```

## 基本文本记忆

MemoryX 的核心功能是提供文本记忆能力，使大型语言模型能够记住对话历史和重要信息。

### 基本用法

```python
from memoryx.core import MemoryController

# 创建记忆控制器
memory = MemoryController()

# 存储记忆
memory.store("用户喜欢蓝色")
memory.store("用户的生日是5月15日")

# 检索记忆
results = memory.search("用户喜欢什么颜色")
print(results)  # 输出: ["用户喜欢蓝色"]

# 存储对话
memory.store_conversation("用户", "你好，我叫张三")
memory.store_conversation("AI", "你好张三，很高兴认识你")

# 获取对话历史
history = memory.get_conversation_history()
print(history)
```

### 高级文本记忆

```python
from memoryx.core import AdvancedMemoryController

# 创建高级记忆控制器
memory = AdvancedMemoryController()

# 存储带标签的记忆
memory.store("用户喜欢蓝色", tags=["偏好", "颜色"])
memory.store("用户的生日是5月15日", tags=["个人信息", "日期"])

# 按标签检索
results = memory.search_by_tags(["偏好"])
print(results)  # 输出: ["用户喜欢蓝色"]

# 设置记忆过期时间（24小时后过期）
memory.store("临时信息", expiry=86400)

# 压缩记忆
summary = memory.compress_memories(["用户喜欢蓝色", "用户讨厌红色", "用户对绿色没有意见"])
print(summary)  # 输出: "用户偏好蓝色，不喜欢红色，对绿色态度中立"
```

## 图像记忆

MemoryX 支持存储和检索图像记忆，使模型能够"记住"和"回忆"图像内容。

### 基本用法

```python
from memoryx.image import ImageMemory
import numpy as np
from PIL import Image

# 创建图像记忆模块
image_memory = ImageMemory()

# 从文件加载图像并存储
image_id = image_memory.store_image("path/to/image.jpg", description="一张海滩照片")

# 或者从 numpy 数组存储
image_array = np.array(Image.open("path/to/image.jpg"))
image_id = image_memory.store_image_array(image_array, description="一张海滩照片")

# 检索图像
image_data = image_memory.retrieve_image(image_id)
print(image_data["description"])  # 输出: "一张海滩照片"

# 搜索相关图像
results = image_memory.search_images("海滩")
for result in results:
    print(result["description"])
```

## 多模态记忆

MemoryX 的多模态功能允许文本和图像记忆的集成，实现更丰富的记忆体验。

### 基本用法

```python
from memoryx.multimodal import MultiModalMemory

# 创建多模态记忆模块
memory = MultiModalMemory()

# 存储文本记忆
text_id = memory.store_text("这是一段关于巴黎的描述")

# 存储图像记忆
image_id = memory.store_image("path/to/paris.jpg")

# 关联文本和图像
memory.associate(text_id, image_id, relationship="描述")

# 检索关联内容
related_items = memory.get_associated_items(text_id)
for item in related_items:
    print(f"类型: {item['type']}, ID: {item['id']}")

# 多模态搜索
results = memory.multimodal_search("巴黎")
print(results)  # 返回相关的文本和图像
```

## 增强图像理解

MemoryX 提供了增强的图像理解能力，包括对象检测、情感分析和高级图像分析。

### 基本用法

```python
from memoryx.enhanced import EnhancedImageUnderstanding

# 创建增强图像理解模块
image_understanding = EnhancedImageUnderstanding()

# 分析图像
analysis = image_understanding.analyze_image("path/to/image.jpg")

# 获取对象检测结果
objects = analysis["objects"]
print(f"检测到 {len(objects)} 个对象")
for obj in objects:
    print(f"对象: {obj['name']}, 置信度: {obj['confidence']}")

# 获取情感分析结果
emotions = analysis["emotions"]
print(f"主要情感: {emotions['primary']}")

# 获取颜色分析
colors = analysis["colors"]
print(f"主要颜色: {colors[0]['name']}")

# 生成详细描述
description = image_understanding.generate_description("path/to/image.jpg")
print(description)
```

## 高级配置

MemoryX 提供了多种配置选项，以适应不同的使用场景。

### 配置文件

您可以创建一个 `config.json` 文件来自定义 MemoryX 的行为：

```json
{
  "memory": {
    "max_items": 1000,
    "default_expiry": 604800,
    "compression_threshold": 50
  },
  "image": {
    "max_resolution": [1024, 1024],
    "storage_format": "jpg",
    "embedding_model": "resnet50"
  },
  "search": {
    "similarity_threshold": 0.7,
    "max_results": 10
  }
}
```

### 代码中配置

```python
from memoryx.core import AdvancedMemoryController

# 创建自定义配置的记忆控制器
memory = AdvancedMemoryController(
    max_items=2000,
    default_expiry=1209600,  # 两周
    compression_threshold=100
)
```

## 常见问题解答

### Q: MemoryX 如何处理大量记忆?

A: MemoryX 使用高效的索引和压缩技术来管理大量记忆。当记忆数量超过阈值时，系统会自动压缩不常用的记忆，并保留重要信息的摘要。

### Q: 图像记忆需要多少存储空间?

A: 默认情况下，MemoryX 会将图像调整为合理的分辨率并进行压缩，一般每张图像需要 100KB-1MB 的存储空间。您可以通过配置文件调整图像存储参数。

### Q: 如何备份记忆数据?

A: MemoryX 的记忆数据默认存储在 `~/.memoryx/` 目录下。您可以定期备份此目录，或使用以下代码导出记忆：

```python
from memoryx.core import MemoryController

memory = MemoryController()
memory.export_memories("backup.json")

# 恢复记忆
memory.import_memories("backup.json")
```

### Q: MemoryX 是否支持分布式部署?

A: 当前版本主要针对单机部署设计。我们计划在未来版本中添加分布式支持，敬请期待。

### Q: 如何清除所有记忆?

A: 使用以下代码清除所有记忆：

```python
from memoryx.core import MemoryController

memory = MemoryController()
memory.clear_all()
```

## 更多示例

请查看 `examples` 目录中的示例脚本，了解更多使用方法：

- `basic_text_memory.py`: 演示基本文本记忆功能
- `image_memory.py`: 演示图像记忆功能
- `enhanced_image_understanding.py`: 演示增强图像理解功能

## 支持与反馈

如果您有任何问题或建议，请在 GitHub 仓库中提交 Issue 或 Pull Request。我们非常欢迎社区贡献！