# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""EnCodec (see https://arxiv.org/abs/2210.13438 and https://arxiv.org/abs/2306.00814)."""

import os
import sys
import torch
import codec_evaluation
import numpy as np
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

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
        model_ckpt_dir=None,
        vocos_ckpt_dir=None,
        need_resample=True,
    ):
        """
        sample_rate: sample rate of the input signal
        orig_sample_rate: original sample rate of the codec
        mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            encode: encode the audio to id tokens
            decode: decode the id tokens to audio
            reconstruct: encode -> decode
            unquantized_emb: encode -> unquantized embedding
            quantized_emb: encode + quantizer -> quantized embedding
        num_codebooks: number of codebooks
        use_vocos: boolean, whether to use Vocos to post-process the audio after decoding
        model_ckpt_dir: path to the model checkpoint
        vocos_ckpt_dir: path to the Vocos checkpoint
        need_resample: boolean, whether to resample the audio after decoding
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
        if model_ckpt_dir is None:
            self.model = EncodecModelHF.from_pretrained(f"facebook/encodec_{tag}khz")
        else:
            self.model = EncodecModelHF.from_pretrained(model_ckpt_dir)
        self.dim = self.model.config.hidden_size
        self.token_rate = self.model.config.frame_rate
        self.hop_length = int(self.orig_sample_rate / self.token_rate)

        self.vocos = None
        if use_vocos:
            try:
                # Workaround to avoid name collisions with installed modules
                root_dir = os.path.dirname(os.path.realpath(__file__))
                sys_path = [x for x in sys.path]
                sys.path = [x for x in sys.path if root_dir not in x]
                self.vocos = self.load_vocos(vocos_ckpt_dir, tag)
                sys.path = sys_path

            except ImportError:
                raise ImportError("`pip install vocos` to use this module")

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

    def load_vocos(self, vocos_ckpt_dir, tag=None):
        from vocos import Vocos

        if vocos_ckpt_dir is None:
            vocos = Vocos.from_pretrained(f"charactr/vocos-encodec-{tag}khz")
        else:
            vocos = Vocos.from_hparams(os.path.join(vocos_ckpt_dir, "config.yaml"))
            vocos.load_state_dict(
                torch.load(
                    os.path.join(vocos_ckpt_dir, "pytorch_model.bin"),
                    map_location="cpu",
                ),
                strict=False,
            )  # Just load vocos parameters, not the model
            vocos.eval()
        return vocos

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
        """
        sig: [B, T]
        return: [B, D, N]    [2, 128, 701]  
        """
        sig, padding_mask = self.process_sig(sig, length)
        unquantized_feats = self.model.encoder(sig)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]    [2, 128, 701]
        """
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(sig, padding_mask, bandwidth=self.bandwidth)
        toks = output.audio_codes[0].movedim(-1, -2)  # [B, N, K]
        quantized_feats = None
        if self.vocos is not None:
            quantized_feats = self.vocos.codes_to_features(toks.long().movedim(-1, 0))
        else:
            quantized_feats = self.model.quantizer.decode(toks.movedim(-1, 0))
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K]    [2, 701, 8]
        """
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            input_values=sig, padding_mask=padding_mask, bandwidth=self.bandwidth
        )
        toks = output.audio_codes[0].movedim(-1, -2)
        return toks, padding_mask

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
        toks: [B, N, K]
        return: [B, T]    [2, 3200]
        """
        if self.vocos is not None:
            bandwidth_id = [1.5, 3.0, 6.0, 12.0].index(self.bandwidth)
            feats = self.vocos.codes_to_features(toks.long().movedim(-1, 0))
            sig = self.vocos.decode(
                feats, bandwidth_id=torch.tensor(bandwidth_id, device=toks.device)
            )  # [B, T]
            return sig
        output = self.model.decode(
            toks[None].movedim(-1, -2), [None], padding_mask=padding_mask
        )
        sig = output.audio_values[:, 0]
        return sig


if __name__ == "__main__":
    import torchaudio

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2
    num_codebooks = 8
    model_ckpt_dir = "/sdb/model_weight/codec_evaluation/codec_ckpt/encodec/models--facebook--encodec_24khz"
    vocos_ckpt_dir = "/sdb/model_weight/codec_evaluation/codec_ckpt/encodec/models--charactr--vocos-encodec-24khz"

    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "example.wav"))
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)  # [B=2, T]

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        for use_vocos in [True, False]:
            codec = (
                Encodec(
                    sample_rate,
                    mode=mode,
                    num_codebooks=num_codebooks,
                    use_vocos=use_vocos,
                    model_ckpt_dir=model_ckpt_dir,
                    vocos_ckpt_dir=vocos_ckpt_dir if use_vocos else None,
                    need_resample=False,  # means the output sample rate is the same as codec's sample rate
                )
                .eval()
                .to(device)
            )
            embs = codec.embs()
            print(
                f"{mode} mode, the codec has {embs.shape[0]} codebooks, use_vocos={use_vocos}, each codebook has {embs.shape[1]} entries, each entry has {embs.shape[2]} dimensions"
            )
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
                save_path = os.path.join(
                    save_dir,
                    f'encodec_reconstruction_{"use_vocos" if use_vocos else "no_vocos"}.wav',
                )
                torchaudio.save(
                    save_path,
                    (
                        output[0].unsqueeze(0).cpu()
                        if use_cuda
                        else output[0].unsqueeze(0)
                    ),
                    codec.orig_sample_rate,
                )
                print(f"{mode} mode has been saved to {save_path}")
            elif mode == "encode":
                print(f"{mode} mode, the output shape is {output[0].shape}")
            else:
                print(f"{mode} mode, the output shape is {output.shape}")
