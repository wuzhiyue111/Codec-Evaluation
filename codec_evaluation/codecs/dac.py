# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""DAC (see https://arxiv.org/abs/2306.06546)."""

import os
import sys
import torch
import torchaudio.transforms as T   
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec


__all__ = ["DAC"]


class DAC(Codec):
    def __init__(self,
        sample_rate,
        orig_sample_rate=24000,
        mode="reconstruct",
        num_codebooks=8,
        need_resample=True,
        model_path: str | None = None
    ):
        """
            sample_rate: sample rate of the input signal
            orig_sample_rate: original sample rate of the codec
            mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            num_codebooks: number of codebooks
            need_resample: Boolean, whether to resample the audio after decoding
            model_path: str, path to the model weights, if None, the model will be downloaded from the internet
        """
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import dac

            sys.path = sys_path
        except ImportError:
            raise ImportError("`pip install descript-audio-codec` to use this module")

        super().__init__(sample_rate, orig_sample_rate, mode)
        self.num_codebooks = num_codebooks
        self.need_resample = need_resample
        self.vocab_size = 1024

        tag = int(orig_sample_rate / 1000)
        if model_path is None:
            model_path = str(dac.utils.download(model_type=f"{tag}khz"))

        self.model = dac.DAC.load(model_path) # model init and load_state_dict
        self.dim = self.model.latent_dim
        self.token_rate = self.model.sample_rate / self.model.hop_length

        # Delete the decoder to save memory overhead.
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None

    # override
    @torch.no_grad()
    def embs(self):
        # H means the dimension of the embedding
        # See https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py#L200
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        toks = (
            toks[:, None, None].expand(-1, self.num_codebooks, -1).clone()
        )  # [C, K, 1] 
        with torch.no_grad():
            z_q, z_p, _ = self.model.quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)  # [C, D, 1] * K
        z_qs = []
        for i, z_p_i in enumerate(z_ps):
            z_q_i = self.model.quantizer.quantizers[i].out_proj(z_p_i)  # [C, H, 1]
            z_qs.append(z_q_i)
        assert (z_q == sum(z_qs)).all()
        # Embeddings pre-projections: size = 8
        # Embeddings post-projections: size = 1024
        embs = torch.stack(z_qs)[..., 0]  # [K, C, H]
        return embs

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]    
        """
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        unquantized_feats = self.model.encoder(sig)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]   
        """
        _, toks, *_ = self.model.encode(
            sig[:, None], n_quantizers = self.num_codebooks
        )   # [B, K, N]
        quantized_feats, _, _ = self.model.quantizer.from_codes(toks)  
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K] 
        """
        _, toks, *_ = self.model.encode(
            sig[:, None], n_quantizers=self.num_codebooks
        )  # tokens.shape = [B, K, N]
        toks = toks.movedim(-1, -2)
        return toks # toks.shape = [B, N, K]

    # override
    def _toks_to_sig(self, toks, length):
        """
            toks: [B, N, K] 
            return: [B, T]
        """
        qfeats, _, _ = self.model.quantizer.from_codes(
            toks.movedim(-1, -2)
        )
        sig = self.model.decode(qfeats)[:, 0] 
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

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            DAC(
                sample_rate,
                mode=mode,
                num_codebooks=num_codebooks,
                need_resample=False, # means the output sample rate is the same as codec's sample rate
                model_path='/sdb/model_weight/codec_evaluation/codec_ckpt/dac_weights_24khz_16kbps_0.0.1.pth'
            )
            .eval()
            .to(device)
        )
        embs = codec.embs()
        print(f'{mode} mode, the codec has {embs.shape[0]} codebooks, each codebook has {embs.shape[1]} entries, each entry has {embs.shape[2]} dimensions')

        if mode == "decode":
            input = torch.zeros(batch_size, 10, num_codebooks).long().to(device)
            with torch.no_grad():
                output = codec(input)
        else:
            with torch.no_grad():
                output = codec(sig)

        if mode == "reconstruct":
            save_dir = os.path.join(root_path, "codecs", "reconstruction_wav")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'dac_reconstruction.wav')
            torchaudio.save(save_path, output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0), codec.orig_sample_rate)
            print(f'{mode} mode has been saved to {save_path}')
        else:
            print(f'{mode} mode, the output shape is {output.shape}')