# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""SpeechTokenizer (see https://arxiv.org/abs/2308.16692)."""

import os
import sys

import torch
from huggingface_hub import snapshot_download

from codec_evaluation.codecs.codec import Codec


__all__ = ["SpeechTokenizer"]


class SpeechTokenizer(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=8,
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import speechtokenizer

            sys.path = sys_path
        except ImportError:
            raise ImportError("`pip install speechtokenizer` to use this module")

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks

        source = "fnlp/SpeechTokenizer"
        path = snapshot_download(repo_id=source)
        config_path = os.path.join(path, "speechtokenizer_hubert_avg", "config.json")
        checkpoint_path = os.path.join(
            path, "speechtokenizer_hubert_avg", "SpeechTokenizer.pt"
        )
        self.model = speechtokenizer.SpeechTokenizer.load_from_checkpoint(
            config_path, checkpoint_path
        )

        # 删除decoder, 节约显存开销
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.transform = None

    # override
    @torch.no_grad()
    def embs(self):
        # See https://github.com/ZhangXInFD/SpeechTokenizer/blob/a9f88dc72642b600654a62861e34342babae6c71/speechtokenizer/quantization/core_vq.py#L360
        vocab_size = 1024
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(vocab_size, device=device)
        toks = (
            toks[None, :, None].expand(self.num_codebooks, -1, -1).clone()
        )  # [K, C, 1]
        embs = []
        for i, indices in enumerate(toks):
            layer = self.model.quantizer.vq.layers[i]
            quantized = layer.decode(indices)  # [C, H, 1]
            embs.append(quantized)
        assert (self.model.quantizer.decode(toks) == sum(embs)).all()
        embs = torch.stack(embs)[..., 0]  # [K, C, H]
        return embs

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        # sig：[B, T]
        unquantized_feats = self.model.sig_to_feats(sig)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        # sig：[B, T]
        toks = self.model.encode(sig[:, None])[: self.num_codebooks]  # [K, B, N]
        quantized_feats = self.model.quantizer(toks)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        toks = self.model.encode(sig[:, None])[: self.num_codebooks]  # [K, B, N]
        toks = toks.movedim(-3, -1)  # [B, N, K]
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        toks = toks.movedim(-1, -3)  # [K, B, N]
        sig = self.model.decode(toks)[:, 0]  # [B, T]
        return sig

# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 8

    # 需要Test
    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            SpeechTokenizer(
                sample_rate,
                mode=mode,
                num_codebooks=num_codebooks,
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
            output = codec(input)
            print(output.shape)
            embs = codec.embs()
            print(embs.shape)

    sig, sample_rate = torchaudio.load("example.wav")
    codec = SpeechTokenizer(sample_rate, num_codebooks=num_codebooks).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, sample_rate)
