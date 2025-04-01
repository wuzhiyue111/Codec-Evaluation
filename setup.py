from setuptools import setup, find_packages
from pathlib import Path


requirements_path = Path(__file__).parent / "requirements.txt"
readme_path = Path(__file__).parent / "README.md"

# 读取 requirements.txt
install_requires = []
with open(requirements_path, encoding="utf-8") as f:
    install_requires.extend([item for item in f.read().splitlines() if item.strip()])

setup(
    name="codec_evaluation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    description="A benchmark for codec evaluation",
    long_description=readme_path.read_text(encoding="utf-8")
    if readme_path.exists()
    else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "codec_eval_probe = codec_evaluation.probe.test.test_inference:cli",
        ]
    },
)
