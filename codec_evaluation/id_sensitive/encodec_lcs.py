from codec_evaluation.init_codecs import init_codec
import torchaudio

def longest_common_substring_strict(a, b):
    """计算整数列表 a 和 b 的最长公共子串（LCSS），要求公共子串的起始位置相同"""
    max_length = 0
    start_index = -1  # 公共子串的起始索引（在a和b中相同）
    
    # 遍历所有可能的起始位置
    for i in range(min(len(a), len(b))):
        current_length = 0
        # 从位置i开始向后匹配，直到元素不同或越界
        while (i + current_length < len(a)) and (i + current_length < len(b)) and (a[i + current_length] == b[i + current_length]):
            current_length += 1
        # 更新最大长度和起始索引
        if current_length > max_length:
            max_length = current_length
            start_index = i
    
    # 提取最长公共子串（如果存在）
    lcs = a[start_index:start_index + max_length] if max_length > 0 else []
    
    return max_length, lcs, start_index, start_index


sig1, sample_rate = torchaudio.load("/home/ch/Codec-Evaluation/codec_evaluation/codecs/example.wav")
encodec1 = init_codec(
    modelname='encodec',
    sample_rate=sample_rate,
    mode="encode",
    num_codebooks=8,
    need_resample=False,
    use_vocos=False,
    vocos_ckpt_dir=None,
    model_ckpt_dir = "/sdb/model_weight/codec_evaluation/codec_ckpt/encodec/models--facebook--encodec_24khz"
)

sig2, sample_rate = torchaudio.load("/home/ch/Codec-Evaluation/codec_evaluation/codecs/2ms_example.wav")
encodec2 = init_codec(
    modelname='encodec',
    sample_rate=sample_rate,
    mode="encode",
    num_codebooks=8,
    need_resample=False,
    use_vocos=False,    
    vocos_ckpt_dir=None,
    model_ckpt_dir = "/sdb/model_weight/codec_evaluation/codec_ckpt/encodec/models--facebook--encodec_24khz"
)

# [1, 8, 701]
i1, _= encodec1(sig1, length=None)
emb1_id = i1.movedim(-1, -2)

i2, _ = encodec2(sig2, length=None)
emb2_id = i2.movedim(-1, -2)

for i in range(emb1_id.shape[1]):
    same = (emb1_id[0, i, :] == emb2_id[0, i, :]).sum().item()
    print(f"codebook{i+1} same: {same}")

# 计算每个 codebook 的最大公共子串及其严格一致的起始索引
result = {}

for i in range(emb1_id.shape[1]):  # 遍历 codebook
    emb1_codebook = emb1_id[0, i, :].tolist()
    emb2_codebook = emb2_id[0, i, :].tolist()
    
    max_length, lcs, start_index_a, start_index_b = longest_common_substring_strict(emb1_codebook, emb2_codebook)
    
    result[f"codebook{i+1}"] = {
        "max_length": max_length,
        "lcs": lcs,
        "start_index_a": start_index_a,
        "start_index_b": start_index_b
    }

# 打印结果
for k, v in result.items():
    print(f"{k}: 最大公共子串长度 {v['max_length']}，为：{v['lcs']}，在 emb1_id 和 emb2_id 中的共同起始索引：{v['start_index_a']}")


