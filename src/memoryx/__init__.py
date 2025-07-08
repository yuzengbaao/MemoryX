"""
MemoryX: 多模态记忆系统

MemoryX是一个为大型语言模型(LLM)设计的多模态记忆系统，旨在解决LLM的"遗忘"问题，
使其能够在长时间的交互中保持上下文连贯性，并提供个性化的用户体验。
"""

__version__ = "0.1.0"

# 导入核心组件
from memoryx.core import MemoryController, AdvancedMemoryController

# 导入图像记忆组件
from memoryx.image import MemoryXImage, ImageProcessor

# 导入多模态组件
from memoryx.multimodal import MemoryXMultimodal, MultimodalMemoryController

# 版本信息
__all__ = [
    "MemoryController",
    "AdvancedMemoryController",
    "MemoryXImage",
    "ImageProcessor",
    "MemoryXMultimodal",
    "MultimodalMemoryController",
]