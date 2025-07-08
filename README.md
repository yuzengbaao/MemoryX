# MemoryX

MemoryX 是一个强大的多模态记忆系统，专为大型语言模型（LLM）设计，提供长期记忆能力。它支持文本和图像记忆，使 LLM 能够"记住"并"回忆"过去的交互和信息。

## 主要特性

- **文本记忆**：存储和检索文本信息，支持对话历史和重要信息的记忆
- **图像记忆**：存储、处理和检索图像，包括生成描述和嵌入
- **多模态集成**：将文本和图像记忆无缝集成，实现更丰富的记忆体验
- **增强图像理解**：提供高级图像分析功能，包括对象检测、情感分析和颜色识别
- **高效搜索**：使用先进的搜索算法快速检索相关记忆
- **记忆压缩**：智能压缩记忆以优化存储空间和检索效率
- **可扩展架构**：模块化设计，易于扩展和定制

## 安装

```bash
# 从 GitHub 安装
git clone https://github.com/yuzengbaao/MemoryX.git
cd MemoryX
pip install -e .

# 从 PyPI 安装 (即将推出)
# pip install memoryx
```

## 快速开始

### 基本文本记忆

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
```

### 图像记忆

```python
from memoryx.image import ImageMemory

# 创建图像记忆模块
image_memory = ImageMemory()

# 存储图像
image_id = image_memory.store_image("path/to/image.jpg", description="一张海滩照片")

# 检索图像
image_data = image_memory.retrieve_image(image_id)
print(image_data["description"])  # 输出: "一张海滩照片"
```

### 增强图像理解

```python
from memoryx.enhanced import EnhancedImageUnderstanding

# 创建增强图像理解模块
image_understanding = EnhancedImageUnderstanding()

# 分析图像
analysis = image_understanding.analyze_image("path/to/image.jpg")
print(f"检测到的对象: {analysis['objects']}")
print(f"主要情感: {analysis['emotions']['primary']}")
```

## 文档

详细的使用说明和 API 文档请参阅 [使用指南](USAGE_GUIDE.md)。

## 示例

查看 `examples` 目录中的示例脚本，了解更多使用方法：

- [基本文本记忆](examples/basic_text_memory.py)
- [图像记忆](examples/image_memory.py)
- [增强图像理解](examples/enhanced_image_understanding.py)

## 贡献

我们欢迎社区贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 致谢

感谢所有为本项目做出贡献的开发者和研究人员。MemoryX 的开发受到了多个开源项目和研究论文的启发。