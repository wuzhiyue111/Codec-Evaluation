from codec_evaluation.init_codecs import init_codec
import torchaudio
import codec_evaluation
import os
import librosa
import numpy as np
import soundfile as sf
import codec_evaluation
import torch
root_path = codec_evaluation.__path__[0]

from codec_evaluation.id_sensitive.utils import (
    longest_common_substring_strict,
)

class IDSensitiveEvaluation:
    def __init__(
        self, 
        codec_name: str,
        model_ckpt_dir: str,
        sample_rate: int,
        mode: str,
        num_codebooks: int,
        need_resample: bool,
        **kwargs,
    ):
        """
        codec_name: codec model name
        model_ckpt_dir: codec model checkpoint directory
        sample_rate: audio sample rate
        mode: codec mode
        num_codebooks: number of codebooks
        need_resample: boolean, whether to resample the audio after decoding
        """
        self.codec = init_codec(
            modelname=codec_name,
            mode=mode,
            sample_rate=sample_rate,
            num_codebooks=num_codebooks,
            need_resample=need_resample,  
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        )
        self.codec_name = codec_name
    
    def get_offset_2ms_audio(self, input_audio_path):
        try:
            # 读取音频
            y, sr = librosa.load(input_audio_path, sr=48000)
            # 计算偏移样本数
            shift_samples = int(sr * 2 / 1000)  # 2ms 对应的样本数

            # 向前移动（右移），前面填充 0
            y_shifted = np.roll(y, shift_samples)
            y_shifted[:shift_samples] = 0  # 避免循环偏移

            # 构建输出音频路径
            output_path = os.path.join(root_path, "id_sensitive", "2ms_example.wav")

            # 保存音频
            sf.write(output_path, y_shifted, sr)
            y_shifted = torch.from_numpy(y_shifted).unsqueeze(0)

            return y_shifted, sr
        except Exception as e:
            print(f"处理音频时出现错误: {e}")
            return None, None

    def get_same_id(self, emb1_id, emb2_id):
        results = []
        for i in range(emb1_id.shape[1]):
            same = (emb1_id[0, i, :] == emb2_id[0, i, :]).sum().item()
            results.append(f"codebook{i+1} same: {same}")
        return "\n".join(results)
 
    def get_lcs(self, emb1_id, emb2_id):
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
        return result
    
    def get_emb(self, sig1, sig2):
        if self.codec_name == "dac":
            i1, _ = self.codec(sig1, length=None)
            emb1_id = self.codec.model.quantizer.from_codes(i1.movedim(-1, -2))[2]
            i2, _ = self.codec(sig2, length=None)
            emb2_id = self.codec.model.quantizer.from_codes(i2.movedim(-1, -2))[2]
        elif self.codec_name == "wavtokenizer":
            i1, _ = self.codec(sig1, length=None)
            emb1_id = i1.movedim(-1, 0)
            i2, _ = self.codec(sig2, length=None)
            emb2_id = i2.movedim(-1, 0)
        else:
            i1, _= self.codec(sig1, length=None)
            emb1_id = i1.movedim(-1, -2)
            i2, _ = self.codec(sig2, length=None)
            emb2_id = i2.movedim(-1, -2)
        return emb1_id, emb2_id
    
    def evaluate(self, task):
        if task == "MRC":
            sig1, sample_rate = torchaudio.load(os.path.join(root_path, "id_sensitive", "example.wav"))
            if sample_rate != args.sample_rate:
                sig1 = torchaudio.functional.resample(sig1, sample_rate, args.sample_rate)
            sig2, _ = torchaudio.load(os.path.join(root_path, "codecs", "reconstruction_wav", f"{self.codec_name}_reconstruction.wav"))

        elif task == "OB2":
            sig1, _ = torchaudio.load(os.path.join(root_path, "id_sensitive", "example.wav"))
            example_audio_path = os.path.join(root_path, "id_sensitive", "example.wav")
            sig2, _ = self.get_offset_2ms_audio(example_audio_path)

        emb1, emb2 = self.get_emb(sig1, sig2)
        same_id_result = self.get_same_id(emb1, emb2)
        lcs_result = self.get_lcs(emb1, emb2)

        print("Same ID Results:")
        print(same_id_result)
        print("LCS Results:")
        for k, v in lcs_result.items():
            print(f"{k}: 最大公共子串长度 {v['max_length']}，为：{v['lcs']}，在 emb1_id 和 emb2_id 中的共同起始索引：{v['start_index_a']}")

        return {
            "same_id": same_id_result,
            "lcs": lcs_result
        }

if __name__ == "__main__":
    """
        if task == "MRC":
            test sample_rate = codec's sample_rate
        elif task == "OB2":
            test sample_rate = example.wav's sample_rate
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="dac")
    parser.add_argument("--model_ckpt_dir", type=str, default="/sdb/model_weight/codec_evaluation/codec_ckpt")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--mode", type=str, default="encode")
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--need_resample", type=bool, default=False)
    parser.add_argument("--task", type=str, default="MRC")
    args = parser.parse_args()

    codec_eval = IDSensitiveEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        sample_rate=args.sample_rate,
        mode=args.mode,
        num_codebooks=args.num_codebooks,
        need_resample=args.need_resample,
    )
    result = codec_eval.evaluate(args.task)
    print(result)
    
    
 
    

    
    
