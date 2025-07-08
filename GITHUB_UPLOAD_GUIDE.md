# GitHub上传指南

本文档提供了将MemoryX项目上传到GitHub的详细步骤。

## 准备工作

1. 创建GitHub账号（如果还没有）
2. 创建新的GitHub仓库
   - 登录GitHub
   - 点击右上角的"+"图标，选择"New repository"
   - 仓库名称填写"memoryx"
   - 添加描述："多模态记忆系统，为大型语言模型提供长期记忆能力"
   - 选择公开（Public）
   - 不要初始化仓库（不勾选"Add a README file"）
   - 点击"Create repository"

## 上传代码

### 方法1：使用命令行

```bash
# 确保你已经在memoryx目录下
cd /path/to/memoryx

# 如果还没有初始化git仓库
git init
git add .
git commit -m "Initial commit: MemoryX多模态记忆系统"

# 添加远程仓库
git remote add origin https://github.com/你的用户名/memoryx.git

# 推送到GitHub
git push -u origin main
```

### 方法2：使用GitHub Desktop

1. 下载并安装[GitHub Desktop](https://desktop.github.com/)
2. 登录你的GitHub账号
3. 添加本地仓库（File > Add local repository）
4. 选择memoryx目录
5. 发布仓库（Repository > Push）

## 验证上传

1. 访问你的GitHub仓库页面：https://github.com/你的用户名/memoryx
2. 确认所有文件都已正确上传
3. 检查README.md是否正确显示

## 后续步骤

1. 设置项目主页
   - 在仓库设置中启用GitHub Pages
   - 选择main分支作为源

2. 添加项目标签
   - 在仓库页面点击"Manage topics"
   - 添加相关标签，如"llm"、"memory"、"multimodal"等

3. 创建发布版本
   - 在仓库页面点击"Releases"
   - 点击"Create a new release"
   - 填写版本号（如v0.1.0）
   - 添加发布说明
   - 点击"Publish release"

4. 设置保护分支
   - 在仓库设置中选择"Branches"
   - 为main分支添加保护规则

## 发布到PyPI（可选）

如果你想让其他人能够通过pip安装你的包，可以按照以下步骤发布到PyPI：

```bash
# 安装打包工具
pip install build twine

# 构建包
python -m build

# 上传到PyPI
twine upload dist/*
```

注意：上传到PyPI前需要在PyPI注册账号，并在setup.py中更新项目URL。