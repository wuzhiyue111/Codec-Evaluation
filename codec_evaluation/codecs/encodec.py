# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""EnCodec (see https://arxiv.org/abs/2210.13438 and https://arxiv.org/abs/2306.00814)."""

import os
import sys
sys.path.append('/home/ch/Codec-Evaluation')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch

from codec_evaluation.codecs.codec import Codec


__all__ = ["Encodec"]


class Encodec(Codec):
    def __init__(
        self,
        sample_rate,
        orig_sample_rate=24000,
        mode="reconstruct",
        num_codebooks=8,
        use_vocos=False,
    ):
        try:
            from transformers import EncodecModel as EncodecModelHF
        except ImportError:
            raise ImportError("`pip install transformers>=4.31.0` to use this module")

        super().__init__(sample_rate, orig_sample_rate, mode)
        self.num_codebooks = num_codebooks
        self.use_vocos = use_vocos
        self.vocab_size = 1024

        tag = int(orig_sample_rate / 1000)
        self.bandwidth = (num_codebooks * 75) / 100
        self.model = EncodecModelHF.from_pretrained(f"facebook/encodec_{tag}khz")
        self.vocos = None
        if use_vocos:
            try:
                # Workaround to avoid name collisions with installed modules
                root_dir = os.path.dirname(os.path.realpath(__file__))
                sys_path = [x for x in sys.path]
                sys.path = [x for x in sys.path if root_dir not in x]
                from vocos import Vocos

                sys.path = sys_path
            except ImportError:
                raise ImportError("`pip install vocos` to use this module")

            self.vocos = Vocos.from_pretrained(f"charactr/vocos-encodec-{tag}khz")
            self.model.decoder = None
        # 删除decoder, 节约显存开销
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None
            self.vocos = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        layers = self.model.quantizer.layers[: self.num_codebooks]
        embs = [layer.codebook.embed for layer in layers]
        embs = torch.stack(embs)  # [K, C, H]
        return embs

    def process_sig(self, sig, length):
        abs_lens = sig.shape[-1] * length
        max_len = abs_lens.max().long().item()
        padding_mask = (
            torch.arange(max_len, device = length.device, dtype = length.dtype)[None, :]
            < abs_lens[:, None]
        )
        return sig[:, None], padding_mask[:, None]

    #override
    # TODO：需要debug，4090debug启动就闪退，需要修理一下
    def _sig_to_unquantized_emb(self, sig, length):
        # sig：[B, T]
        sig, padding_mask = self.process_sig(sig, length)
        # 传递 sig 和 padding_mask 为字典形式，符合 transformers 模型的输入要求
        input_data = {"input_ids": sig, "attention_mask": padding_mask}
        output = self.model.encoder(input_data)
        unquantized_feats = output.get("hidden_states")
        if unquantized_feats is None:
            raise ValueError("unquantized_feats not found in encoder output")
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        # sig：[B, T]
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, bandwidth = self.bandwidth
        )
        toks = output.audio_codes[0].movedim(-1, -2)    # [B, N, K]
        if self.vocos is not None:
            bandwidth_id = [1.5, 3.0, 6.0, 12.0].index(self.bandwidth)
            quantized_feats = self.vocos.codes_to_features(toks.long().movedim(-1, 0))
            return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, bandwidth=self.bandwidth
        )
        toks = output.audio_codes[0].movedim(-1, -2)  # [B, N, K]
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        if self.vocos is not None:
            bandwidth_id = [1.5, 3.0, 6.0, 12.0].index(self.bandwidth)
            feats = self.vocos.codes_to_features(toks.long().movedim(-1, 0))
            sig = self.vocos.decode(
                feats, bandwidth_id=torch.tensor(bandwidth_id, device=toks.device)
            )  # [B, T]
            return sig
        output = self.model.decode(toks[None].movedim(-1, -2), [None])
        sig = output.audio_values[:, 0]  # [B, T]
        return sig

# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 8

    # 需要Test
    # encodec：sig to unquantized emb有一点bug，但是4090的debug突然不好使了，在处理，其他输出正常；
    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        for use_vocos in [False, True]:
            codec = (
                Encodec(
                    sample_rate,
                    mode=mode,
                    num_codebooks=num_codebooks,
                    use_vocos=use_vocos,
                )
                .eval()
                .to(device)
            )
            input = (
                torch.zeros(batch_size, 10, num_codebooks).long()
                if mode == "decode"
                else torch.randn(batch_size, sample_rate)
            ).to(device)
            with torch.no_grad():
                # output = codec(input)
                # print(output.shape)

                output = codec(input)
                if output is not None:
                    print(output.shape)
                else:
                    print("错误：codec 输出为 None。")

                embs = codec.embs()
                print(embs.shape)

    sig, sample_rate = torchaudio.load("/home/ch/Codec-Evaluation/example_audio/encodec/vctk_p225_012.wav")
    codec = Encodec(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("/home/ch/Codec-Evaluation/reconstruction_audio/encodec/vctk_reconstruction.wav", rec_sig, sample_rate)
