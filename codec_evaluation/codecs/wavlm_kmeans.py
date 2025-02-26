# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""WavLM + K-means (see https://arxiv.org/abs/2312.09747)."""
import sys
import torch
import codec_evaluation
path_root = codec_evaluation.__path__[0]
sys.path.append(path_root)
from codec_evaluation.codecs.codec import Codec


__all__ = ["WavLMKmeans"]


class WavLMKmeans(Codec):
    LAYER_IDS = [(6,), (1, 3, 6)]

    def __init__(self, sample_rate, mode="reconstruct", layer_ids=(6,), need_resample=True):
        super().__init__(sample_rate, 16000, mode)
        self.layer_ids = layer_ids
        self.vocab_size = 512

        self.model = torch.hub.load(
            repo_or_dir="lucadellalib/discrete-wavlm-codec",
            model="discrete_wavlm_large",
            layer_ids=layer_ids,
        )   
        self.need_resample = need_resample
        self.dim = self.model.vocoder.embedding_dim

        # Delete the decoder to save memory overhead.
        if mode == "encode" or mode == "unquantized_emb":
            self.model.dequantizer = None
            self.model.vocoder = None
        elif mode == "quantized_emb":
            self.model.vocoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        toks = toks[:, None].expand(-1, len(self.layer_ids)).clone()  # [C, K]
        embs = self.model.toks_to_qfeats(toks)  # [C, H, K]
        embs = embs.movedim(-1, 0)  # [K, C, H] = [1, 512, 1024]
        return embs

    # override 
    """
        sig: [B, T] 
        unquantized_feats: [B, D, N] 
    """
    def _sig_to_unquantized_emb(self, sig, length):
        unquantized_feats = self.model.sig_to_feats(sig).mean(dim=-1).movedim(-1, -2)
        return unquantized_feats

    # override
    """
        sig: [B, T]
        quantized_feats: [B, D, N] 
    """
    def _sig_to_quantized_emb(self, sig, length):
        toks = self._sig_to_toks(sig, length)
        quantized_feats = self.model.toks_to_qfeats(toks).mean(dim=-1).movedim(-1, -2)
        return quantized_feats

    # override
    """
        sig: [B, T]
        toks: [B, N, K] 
    """
    def _sig_to_toks(self, sig, length):
        feats = self.model.sig_to_feats(sig)
        toks = self.model.feats_to_toks(feats)  
        return toks

    # override
    """
        toks: [B, N, K]
        sig: [B, T]
    """
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        quantized_feats = self.model.toks_to_qfeats(toks)
        feats = self.model.qfeats_to_feats(quantized_feats)
        sig = self.model.feats_to_sig(feats)[:, 0]  
        return sig


if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    layer_ids = [6]

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            WavLMKmeans(sample_rate, mode=mode, layer_ids=layer_ids).eval().to(device)
        )
        input = (
            torch.zeros(batch_size, 10, len(layer_ids)).long()
            if mode == "decode"
            else torch.randn(batch_size, sample_rate)
        ).to(device)
        with torch.no_grad():
            output = codec(input)
            print(output.shape)
            embs = codec.embs()
            print(embs.shape)

    sig, sample_rate = torchaudio.load("example.wav")
    codec = WavLMKmeans(sample_rate, layer_ids=layer_ids, need_resample=False).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, codec.orig_sample_rate)