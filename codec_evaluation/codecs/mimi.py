# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Mimi (see https://kyutai.org/Moshi.pdf)."""
import sys
import torch
import codec_evaluation
path_root = codec_evaluation.__path__[0]
sys.path.append(path_root)

from codec_evaluation.codecs.codec import Codec


__all__ = ["Mimi"]


class Mimi(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=8,
    ):
        try:
            from transformers import MimiModel
        except ImportError:
            raise ImportError("`pip install transformers>=4.45.1` to use this module")

        super().__init__(sample_rate, 24000, mode)
        self.num_codebooks = num_codebooks
        self.vocab_size = 2048

        self.model = MimiModel.from_pretrained("kyutai/mimi")
        
        # 删除decoder, 节约显存开销
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None
            self.model.decoder_transformer = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.encoder_transformer = None

    # override
    @torch.no_grad()
    def embs(self):
        semantic_layers = self.model.quantizer.semantic_residual_vector_quantizer.layers
        acoustic_layers = self.model.quantizer.acoustic_residual_vector_quantizer.layers
        layers = (semantic_layers + acoustic_layers)[: self.num_codebooks]
        embs = [layer.codebook.embed for layer in layers]
        embs = torch.stack(embs)  # [K, C, H]
        return embs

    def process_sig(self, sig, length):
        # sig: [B, T]
        abs_lens = sig.shape[-1] * length
        max_len = abs_lens.max().long().item()
        padding_mask = (
            torch.arange(max_len, device=length.device, dtype=length.dtype)[None, :]
            < abs_lens[:, None]
        )
        return sig[:, None], padding_mask[:, None]

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        # sig: [B, T]
        sig, padding_mask = self.process_sig(sig, length)
        unquantized_feats = self.model.encoder(sig)
        return unquantized_feats

    # override
    # 修改了MimiSplitResidualVectorQuantizer方法，添加了forward函数
    def _sig_to_quantized_emb(self, sig, length):
        # sig: [B, T]
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, num_quantizers=self.num_codebooks
        )
        toks = output.audio_codes.movedim(-1, -2)  # [B, N, K]
        quantized_feats = self.model.quantizer(
            toks.movedim(-1, -2)    # [B, K, N]
        )
        return quantized_feats
    
    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, num_quantizers=self.num_codebooks
        )
        toks = output.audio_codes.movedim(-1, -2)  # [B, N, K]
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        output = self.model.decode(toks.movedim(-1, -2))
        sig = output.audio_values[:, 0]  # [B, T]
        return sig


if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 8
    
    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            Mimi(sample_rate, mode=mode, num_codebooks=num_codebooks).eval().to(device)
        )
        input = (
            torch.zeros(batch_size, 10, num_codebooks).long()
            if mode == "decode"
            else torch.randn(batch_size, sample_rate)
        ).to(device)
        with torch.no_grad():
            output = codec(input)
            if output is not None:
                print("codec(input):" + str(output.shape))
            else:
                print("错误：codec 输出为 None。")
            embs = codec.embs()
            print("emb.shape:" + str(embs.shape))

    sig, sample_rate = torchaudio.load("example.wav")
    codec = Mimi(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruct.wav", rec_sig, sample_rate)
