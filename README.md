# Codec-Evaluation

### Purpose
1. 如何判断codebook学习的好坏
2. 集合所有 codec 现有的评测指标

### Env Build
conda create -n codec_eval python==3.10 -y
conda activate codec_eval

git clone https://github.com/wuzhiyue111/Codec-Evaluation.git
cd Codec-Evaluation

bash env_build.sh

### Road Map
- [ ] 不同 codec 的封装
- [ ] 不同数据集的清洗
- [ ] Codec 评测指标规则的制定
