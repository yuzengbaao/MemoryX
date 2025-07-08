#!/bin/bash
# MemoryX 项目自动上传脚本

echo "===== MemoryX 项目自动上传工具 ====="
echo ""

# 检查是否安装了必要的工具
command -v git >/dev/null 2>&1 || { echo "错误: 未安装git"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "错误: 未安装python3"; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "警告: 未安装pip，将无法安装依赖"; }

# 安装依赖
echo "正在安装必要的依赖..."
pip install requests >/dev/null 2>&1 || { echo "警告: 无法安装requests库，但将继续尝试"; }

# 检查是否已设置GitHub Token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "未检测到GITHUB_TOKEN环境变量"
    echo "请输入您的GitHub个人访问令牌 (Personal Access Token):"
    read -s GITHUB_TOKEN
    export GITHUB_TOKEN
    
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "错误: 未提供GitHub Token"
        exit 1
    fi
else
    echo "已检测到GITHUB_TOKEN环境变量"
fi

# 执行上传脚本
echo ""
echo "开始执行上传过程..."
python3 upload_to_github.py

# 检查上传结果
if [ $? -eq 0 ]; then
    echo ""
    echo "上传过程已完成!"
    echo "请查看上方输出的仓库URL以访问您的项目"
else
    echo ""
    echo "上传过程中遇到错误，请查看上方错误信息"
    echo "您也可以尝试手动上传，详见 upload_instructions.md"
fi