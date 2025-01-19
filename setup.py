from setuptools import setup, find_packages
import os

# 读取 requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="codec_evaluation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    description="A benchmark for codec evaluation",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    python_requires=">=3.10",
)
