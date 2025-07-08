#!/usr/bin/env python3
"""
自动化脚本，用于创建GitHub仓库并上传MemoryX项目
"""

import os
import sys
import subprocess
import requests
import json
import time

def run_command(command):
    """执行shell命令并返回输出"""
    print(f"执行命令: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"错误: {stderr.decode('utf-8')}")
        return False, stderr.decode('utf-8')
    
    return True, stdout.decode('utf-8')

def create_github_repo(token, repo_name, description):
    """创建GitHub仓库"""
    print(f"正在创建GitHub仓库: {repo_name}")
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    data = {
        'name': repo_name,
        'description': description,
        'private': False,
        'has_issues': True,
        'has_projects': True,
        'has_wiki': True
    }
    
    response = requests.post(
        'https://api.github.com/user/repos',
        headers=headers,
        data=json.dumps(data)
    )
    
    if response.status_code == 201:
        print("仓库创建成功!")
        return True, response.json()
    else:
        print(f"创建仓库失败: {response.status_code}")
        print(response.text)
        return False, response.text

def push_to_github(token, username, repo_name):
    """推送代码到GitHub"""
    print(f"正在推送代码到GitHub: {username}/{repo_name}")
    
    # 设置远程仓库
    remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    success, output = run_command(f"git remote add origin {remote_url}")
    if not success:
        # 如果已存在，尝试设置URL
        run_command(f"git remote set-url origin {remote_url}")
    
    # 推送代码
    success, output = run_command("git push -u origin main")
    if success:
        print("代码推送成功!")
        return True
    else:
        print("代码推送失败!")
        return False

def main():
    """主函数"""
    # 获取GitHub Token
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        token = input("请输入GitHub Token: ")
        if not token:
            print("错误: 未提供GitHub Token")
            sys.exit(1)
    
    # 获取GitHub用户名
    headers = {'Authorization': f'token {token}'}
    response = requests.get('https://api.github.com/user', headers=headers)
    if response.status_code != 200:
        print(f"错误: 无法获取GitHub用户信息 ({response.status_code})")
        print(response.text)
        sys.exit(1)
    
    username = response.json()['login']
    print(f"GitHub用户名: {username}")
    
    # 仓库信息
    repo_name = "memoryx"
    description = "多模态记忆系统，为大型语言模型提供长期记忆能力"
    
    # 创建仓库
    success, repo_info = create_github_repo(token, repo_name, description)
    if not success:
        # 检查是否是因为仓库已存在
        if "name already exists" in repo_info:
            print("仓库已存在，将直接推送代码")
        else:
            print("错误: 无法创建GitHub仓库")
            sys.exit(1)
    
    # 等待仓库创建完成
    print("等待仓库准备就绪...")
    time.sleep(3)
    
    # 推送代码
    success = push_to_github(token, username, repo_name)
    if not success:
        print("错误: 无法推送代码到GitHub")
        sys.exit(1)
    
    # 输出成功信息
    print("\n" + "="*50)
    print(f"MemoryX项目已成功上传到GitHub!")
    print(f"仓库地址: https://github.com/{username}/{repo_name}")
    print("="*50)

if __name__ == "__main__":
    main()