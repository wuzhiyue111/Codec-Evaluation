from codec_evaluation.init_codecs import init_codec
import torchaudio
import codec_evaluation
import os
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
            sample_rate=sample_rate,
            mode=mode,
            num_codebooks=num_codebooks,
            need_resample=need_resample,  
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        )
        self.codec_name = codec_name

    def get_same_id(self, emb1_id, emb2_id):
        for i in range(emb1_id.shape[1]):
            same = (emb1_id[0, i, :] == emb2_id[0, i, :]).sum().item()
            print(f"codebook{i+1} same: {same}")
 
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

        # 打印结果
        for k, v in result.items():
            print(f"{k}: 最大公共子串长度 {v['max_length']}，为：{v['lcs']}，在 emb1_id 和 emb2_id 中的共同起始索引：{v['start_index_a']}")
    
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
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="dac")
    parser.add_argument("--model_ckpt_dir", type=str, default="/sdb/model_weight/codec_evaluation/codec_ckpt")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--mode", type=str, default="encode")
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--need_resample", type=bool, default=False)
    parser.add_argument("--task", type=str, default="MRC")
    args = parser.parse_args()

    """ 
        TODO:MRC任务还需要思考一下
        task: MRC   Multi-round Refactoring Comparison
              OB2   Offset by 2ms
    """
    if args.task == "MRC":
        sig1, sample_rate = torchaudio.load(os.path.join(root_path, "id_sensitive", "example.wav"))
        sig2, sample_rate = torchaudio.load(os.path.join(root_path, "id_sensitive", f"{args.codec_name}_reconstruction.wav"))
    else:
        sig1, sample_rate = torchaudio.load(os.path.join(root_path, "id_sensitive", "example.wav"))
        sig2, sample_rate = torchaudio.load(os.path.join(root_path, "id_sensitive", "2ms_example.wav"))

    codec_eval = IDSensitiveEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        sample_rate=args.sample_rate,
        mode=args.mode,
        num_codebooks=args.num_codebooks,
        need_resample=args.need_resample,
    )

    emb1, emb2 = codec_eval.get_emb(sig1, sig2)
    codec_eval.get_same_id(emb1, emb2)
    codec_eval.get_lcs(emb1, emb2)
 
    

    
    
