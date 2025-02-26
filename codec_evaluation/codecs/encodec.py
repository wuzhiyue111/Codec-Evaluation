# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""EnCodec (see https://arxiv.org/abs/2210.13438 and https://arxiv.org/abs/2306.00814)."""

import os
import sys
import torch
import codec_evaluation
path_root = codec_evaluation.__path__[0]
sys.path.append(path_root)

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
        need_resample=True,
    ):
        """
            sample_rate: sample rate of the input signal
            orig_sample_rate: original sample rate of the codec
            mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            num_codebooks: number of codebooks
            use_vocos: Boolean, whether to use Vocos to post-process the audio after decoding
            need_resample: Boolean, whether to resample the audio after decoding
        """
        try:
            from transformers import EncodecModel as EncodecModelHF
        except ImportError:
            raise ImportError("`pip install transformers>=4.31.0` to use this module")

        super().__init__(sample_rate, orig_sample_rate, mode)
        self.num_codebooks = num_codebooks
        self.use_vocos = use_vocos
        self.need_resample = need_resample
        self.vocab_size = 1024

        tag = int(orig_sample_rate / 1000)
        self.bandwidth = (num_codebooks * 75) / 100
        self.model = EncodecModelHF.from_pretrained(f"facebook/encodec_{tag}khz")
        self.dim = self.model.config.hidden_size
        
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
        # Delete the decoder to save memory overhead.
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
        # sig: [B, T]
        abs_lens = sig.shape[-1] * length
        max_len = abs_lens.max().long().item()
        padding_mask = (
            torch.arange(max_len, device = length.device, dtype = length.dtype)[None, :]
            < abs_lens[:, None]
        )
        return sig[:, None], padding_mask[:, None]

    #override
    """
        sig: [B, T]
        unquantized_feats: [B, D, N] 
    """
    def _sig_to_unquantized_emb(self, sig, length):
        sig, padding_mask = self.process_sig(sig, length)
        unquantized_feats = self.model.encoder(sig)
        return unquantized_feats

    # override
    """
        sig: [B, T]
        quantized_feats: [B, D, N] 
    """
    def _sig_to_quantized_emb(self, sig, length):
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, bandwidth=self.bandwidth
        )
        toks = output.audio_codes[0].movedim(-1, -2)    # [B, N, K]
        quantized_feats = None
        if self.vocos is not None:
            quantized_feats = self.vocos.codes_to_features(toks.long().movedim(-1, 0))
        else:
            quantized_feats = self.model.quantizer.decode(toks.movedim(-1, 0))
        return quantized_feats

    # override
    """
        sig: [B, T]
        toks: [B, N, K] 
    """
    def _sig_to_toks(self, sig, length):
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, bandwidth=self.bandwidth
        )
        toks = output.audio_codes[0].movedim(-1, -2)  
        return toks

    # override
    """
        toks: [B, N, K]
        sig: [B, T]
    """
    def _toks_to_sig(self, toks, length):
        if self.vocos is not None:
            bandwidth_id = [1.5, 3.0, 6.0, 12.0].index(self.bandwidth)
            feats = self.vocos.codes_to_features(toks.long().movedim(-1, 0))
            sig = self.vocos.decode(
                feats, bandwidth_id=torch.tensor(bandwidth_id, device=toks.device)
            )  # [B, T] 
            return sig
        output = self.model.decode(toks[None].movedim(-1, -2), [None])
        sig = output.audio_values[:, 0]  
        return sig

if __name__ == "__main__":
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 10000
    batch_size = 2
    num_codebooks = 8

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        for use_vocos in [True, False]:
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
                output = codec(input)
                print(output.shape)
                embs = codec.embs()
                print(embs.shape)

    sig, sample_rate = torchaudio.load("example.wav")
    codec = Encodec(sample_rate, num_codebooks=num_codebooks, need_resample=False).eval()
    with torch.no_grad():
        rec_sig = codec(sig)
    torchaudio.save("reconstruction.wav", rec_sig, codec.orig_sample_rate)