# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""WavTokenizer (see https://arxiv.org/abs/2408.16532)."""

import os
import sys
sys.path.append('/home/ch/Codec-Evaluation')

import torch
from huggingface_hub import snapshot_download

from codec_evaluation.codecs.codec import Codec


__all__ = ["WavTokenizer"]


class WavTokenizer(Codec):
    SOURCES = [
        "novateur/WavTokenizer-large-unify-40token",
        "novateur/WavTokenizer-large-speech-75token",
    ]
    CONFIGS = [
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    ]
    CHECKPOINTS = [
        "wavtokenizer_large_unify_600_24k.ckpt",
        "wavtokenizer_large_speech_320_24k.ckpt",
    ]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        source="novateur/WavTokenizer-large-unify-40token",
        config="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        checkpoint="wavtokenizer_large_unify_600_24k.ckpt",
    ):
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import wavtokenizer

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/lucadellalib/WavTokenizer.git@main` to use this module"
            )

        super().__init__(sample_rate, 24000, mode)
        self.num_codebooks = 1
        self.vocab_size = 4096

        path = snapshot_download(repo_id=source)
        checkpoint_path = os.path.join(path, checkpoint)
        path = snapshot_download(repo_id="novateur/WavTokenizer")
        config_path = os.path.join(path, config)
        self.model = wavtokenizer.WavTokenizer.from_pretrained0802(
            config_path, checkpoint_path
        )

        # 删除decoder, 节约显存开销
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.feature_extractor.encodec.decoder = None
            self.model.head = None
        elif mode == "decode":
            self.model.feature_extractor.encodec.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        embs = self.model.feature_extractor.encodec.quantizer.vq.layers[0].codebook
        embs = embs[None]  # [K, C, H]
        return embs

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        # sig: [B, T]
        unquantized_feats = self.model.sig_to_feats(sig)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        # sig：[B, T]
        toks = self._sig_to_toks(sig, length)
        quantized_feats = self.model.codes_to_features(toks.movedim(-1, 0))
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        _, toks = self.model.encode(sig, bandwidth_id=0)
        toks = toks.movedim(0, -1)  # [B, N, K]
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        quantized_feats = self.model.codes_to_features(toks.movedim(-1, 0))
        sig = self.model.decode(
            quantized_feats, bandwidth_id=torch.tensor(0, device=toks.device)
        )  # [B, T]
        return sig

# Test
if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2

    # 需要Test
    # wavtokenizer：github超时
    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = WavTokenizer(sample_rate, mode=mode).eval().to(device)
        input = (
            torch.zeros(batch_size, 10, 1).long()
            if mode == "decode"
            else torch.randn(batch_size, sample_rate)
        ).to(device)
        with torch.no_grad():
            output = codec(input)
            print(output.shape)
            embs = codec.embs()
            print(embs.shape)

    sig, sample_rate = torchaudio.load("/home/ch/Codec-Evaluation/example_audio/wavtokenizer/vctk_p225_018.wav")
    codec = WavTokenizer(sample_rate).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("/home/ch/Codec-Evaluation/reconstruction_audio/wavtokenizer/vctk_reconstruction.wav", rec_sig, sample_rate)
