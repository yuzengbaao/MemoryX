from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memoryx",
    version="0.1.0",
    author="MemoryX Contributors",
    author_email="memoryx@example.com",
    description="多模态记忆系统，为大型语言模型提供长期记忆能力",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/memoryx",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/memoryx/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "flake8>=3.8.4",
            "mypy>=0.800",
        ],
        "enhanced": [
            "torch>=1.7.0",
            "transformers>=4.5.0",
            "networkx>=2.5",
        ],
    },
)