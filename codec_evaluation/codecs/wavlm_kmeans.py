# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""WavLM + K-means (see https://arxiv.org/abs/2312.09747)."""
import sys
import torch
import codec_evaluation
import os
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)
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
    def _sig_to_unquantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N] 
        """
        unquantized_feats = self.model.sig_to_feats(sig).mean(dim=-1).movedim(-1, -2)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N] 
        """
        toks = self._sig_to_toks(sig, length)
        quantized_feats = self.model.toks_to_qfeats(toks).mean(dim=-1).movedim(-1, -2)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K] 
        """
        feats = self.model.sig_to_feats(sig)
        toks = self.model.feats_to_toks(feats)  
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        """
            toks: [B, N, K]
            return: [B, T]
        """
        quantized_feats = self.model.toks_to_qfeats(toks)
        feats = self.model.qfeats_to_feats(quantized_feats)
        sig = self.model.feats_to_sig(feats)[:, 0]  
        return sig


if __name__ == "__main__":
    import torchaudio
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2
    num_codebooks = 8
    
    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "example.wav"))
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1) # [B=2, T]
    layer_ids = [6]

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            WavLMKmeans(sample_rate, mode=mode, layer_ids=layer_ids).eval().to(device)
        )
        embs = codec.embs()
        print(f'{mode} mode, the codec has {embs.shape[0]} codebooks, each codebook has {embs.shape[1]} entries, each entry has {embs.shape[2]} dimensions')
        if mode == "decode":
            input = torch.zeros(batch_size, 10, len(layer_ids)).long().to(device)
            with torch.no_grad():
                output = codec(input)
        else:
            with torch.no_grad():
                output = codec(sig)

        if mode == "reconstruct":
            save_dir = os.path.join(root_path, "codecs", "reconstruction_wav")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'wavlm_kmeans_reconstruction.wav')
            torchaudio.save(save_path, output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0), codec.orig_sample_rate)
            print(f'{mode} mode has been saved to {save_path}')
        else:
            print(f'{mode} mode, the output shape is {output.shape}')