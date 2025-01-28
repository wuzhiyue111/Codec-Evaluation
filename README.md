# Codec-Evaluation

## Purpose

1. how to evaluate the quality of codebook
2. collect all existing metrics

## Env Build

conda create -n codec_eval python==3.10 -y

conda activate codec_eval

git clone https://github.com/wuzhiyue111/Codec-Evaluation.git

cd Codec-Evaluation

bash env_build.sh

## Road Map

- [ ] multi codec deploy
    - multi codec deploy reference: https://github.com/lucadellalib/audiocodecs
- [ ] clean different dataset in marble benchmark
    - add code to redeploy marble in our benchmark
    - add code in marble base to evaluate our index 
- [ ] define the evaluation metrics of codec, codebooks
    - test ID sensitive in same semantic
    - 




