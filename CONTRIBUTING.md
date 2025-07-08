# 贡献指南

感谢您对MemoryX项目的关注！我们欢迎各种形式的贡献，包括但不限于代码贡献、文档改进、问题报告和功能建议。

## 如何贡献

### 报告问题

如果您发现了bug或有功能请求，请通过GitHub Issues提交。提交问题时，请尽可能详细地描述：

1. 问题的具体表现或您希望的功能
2. 复现步骤（如适用）
3. 预期行为与实际行为
4. 环境信息（操作系统、Python版本等）
5. 相关的日志或截图

### 提交代码

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个Pull Request

### 代码风格

我们使用以下代码风格规范：

- 遵循PEP 8风格指南
- 使用4个空格进行缩进（不使用制表符）
- 使用有意义的变量名和函数名
- 为函数和类添加文档字符串
- 保持代码简洁明了

### 测试

在提交代码之前，请确保：

1. 添加适当的单元测试
2. 所有测试都能通过
3. 代码覆盖率不会降低

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/memoryx.git
cd memoryx

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"
```

## 项目结构

```
memoryx/
├── src/
│   ├── memoryx/
│   │   ├── __init__.py
│   │   ├── core/           # 核心记忆模块
│   │   ├── image/          # 图像记忆模块
│   │   ├── multimodal/     # 多模态集成模块
│   │   └── enhanced/       # 增强型图像理解模块
├── tests/                  # 测试目录
├── examples/               # 示例代码
├── docs/                   # 文档
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── setup.py
└── pyproject.toml
```

## 发布流程

1. 更新版本号（遵循语义化版本规范）
2. 更新CHANGELOG.md
3. 创建一个新的发布标签
4. 构建并上传到PyPI

## 行为准则

请参阅我们的[行为准则](CODE_OF_CONDUCT.md)，以了解我们的社区标准。

## 许可证

通过贡献您的代码，您同意您的贡献将根据项目的[MIT许可证](LICENSE)进行许可。