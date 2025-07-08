# MemoryX 项目上传指南

本文档提供了两种将 MemoryX 项目上传到 GitHub 的方法：自动化脚本和手动步骤。

## 方法一：使用自动化脚本（推荐）

我们提供了一个自动化脚本，可以帮助您快速创建 GitHub 仓库并上传代码。

### 前提条件

1. 您需要有一个 GitHub 账号
2. 您需要创建一个 GitHub 个人访问令牌 (Personal Access Token)
   - 访问 https://github.com/settings/tokens
   - 点击 "Generate new token"
   - 选择 "repo" 权限
   - 生成并复制令牌

### 执行步骤

1. 设置环境变量（推荐）：
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```
   
   或者运行脚本时手动输入令牌。

2. 运行上传脚本：
   ```bash
   cd /workspace/memoryx
   ./upload_to_github.py
   ```

3. 脚本将自动：
   - 创建名为 "memoryx" 的 GitHub 仓库
   - 将本地代码推送到该仓库
   - 输出仓库 URL

### 故障排除

如果遇到问题，请检查：
- GitHub 令牌是否有效
- 是否已经存在同名仓库
- 网络连接是否正常

## 方法二：手动上传

如果您更喜欢手动控制上传过程，可以按照以下步骤操作：

1. 在 GitHub 上创建新仓库：
   - 访问 https://github.com/new
   - 仓库名称填写 "memoryx"
   - 添加描述："多模态记忆系统，为大型语言模型提供长期记忆能力"
   - 选择公开 (Public)
   - 不要初始化仓库（不勾选 "Add a README file"）
   - 点击 "Create repository"

2. 在本地配置远程仓库：
   ```bash
   cd /workspace/memoryx
   git remote add origin https://github.com/您的用户名/memoryx.git
   ```

3. 推送代码：
   ```bash
   git push -u origin main
   ```

## 上传后的后续步骤

无论使用哪种方法，成功上传后您可以：

1. 设置项目主页：
   - 在仓库设置中启用 GitHub Pages
   - 选择 main 分支作为源

2. 添加项目标签：
   - 在仓库页面点击 "Manage topics"
   - 添加相关标签，如 "llm"、"memory"、"multimodal" 等

3. 创建发布版本：
   - 在仓库页面点击 "Releases"
   - 点击 "Create a new release"
   - 填写版本号（如 v0.1.0）
   - 添加发布说明
   - 点击 "Publish release"

4. 发布到 PyPI（可选）：
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```

## 注意事项

- 确保您的 GitHub 账号有足够的权限创建仓库
- 如果您计划将此项目作为开源项目，请确保遵循 MIT 许可证的要求
- 定期更新代码并推送到 GitHub 以保持项目活跃