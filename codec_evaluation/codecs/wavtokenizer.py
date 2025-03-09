# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""WavTokenizer (see https://arxiv.org/abs/2408.16532)."""

import os
import sys
import torch
import codec_evaluation

root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

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
        need_resample=True,
        mode="reconstruct",
        model_ckpt_dir=None,
        source="novateur/WavTokenizer-large-unify-40token",
        config="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        checkpoint="wavtokenizer_large_unify_600_24k.ckpt",
    ):
        """
        sample_rate: sample rate of the input signal
        need_resample: Boolean, whether to resample the audio after decoding
        mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            encode: encode the audio to id tokens
            decode: decode the id tokens to audio
            reconstruct: encode -> decode
            unquantized_emb: encode -> unquantized embedding
            quantized_emb: encode + quantizer -> quantized embedding
        source: source of the model
        config: config of the model
        checkpoint: checkpoint of the model
        """
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
        self.need_resample = need_resample
        self.num_codebooks = 1
        self.vocab_size = 4096

        if model_ckpt_dir is None:
            path = snapshot_download(repo_id=source)
            checkpoint_path = os.path.join(path, checkpoint)
            path = snapshot_download(repo_id="novateur/WavTokenizer")
            config_path = os.path.join(path, config)
        else:
            checkpoint_path = os.path.join(model_ckpt_dir, checkpoint)
            config_path = os.path.join(model_ckpt_dir, config)
        self.model: wavtokenizer.WavTokenizer = wavtokenizer.WavTokenizer.from_pretrained0802(
            config_path, checkpoint_path
        )
        self.hop_length = self.model.feature_extractor.encodec.encoder.hop_length
        self.dim = self.model.feature_extractor.encodec.encoder.dimension
        self.token_rate = self.model.feature_extractor.encodec.frame_rate

        # Delete the decoder to save memory overhead.
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
        """
            sig: [B, T]
            return: [B, D, N]   [2, 512, 6924]
        """
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        unquantized_feats = self.model.feature_extractor.encodec.encoder(sig)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]   [2, 512, 6924]
        """
        quantized_feats, _ = self.model.encode(sig, bandwidth_id=0)
        quantized_feats = quantized_feats.clone()
        quantized_feats.requires_grad_(True)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
        sig: [B, T]
        return: [B, N, K]   [2, 6924, 1]
        """
        _, toks = self.model.encode(sig, bandwidth_id=0)
        toks = toks.movedim(0, -1)
        return toks, None

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
        toks: [B, N, K]
        return: [B, T]  [2, 6000]
        """
        quantized_feats = self.model.codes_to_features(toks.movedim(-1, 0))
        sig = self.model.decode(
            quantized_feats, bandwidth_id=torch.tensor(0, device=toks.device)
        )
        return sig


if __name__ == "__main__":
    import torchaudio

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2
    num_codebooks = 8

    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "example.wav"))
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)  # [B=2, T]

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            WavTokenizer(
                sample_rate,
                mode=mode,
                model_ckpt_dir="/sdb/model_weight/codec_evaluation/codec_ckpt/wavTokenizer/models--novateur--WavTokenizer-large-unify-40token",
                need_resample=False,
            )
            .eval()
            .to(device)
        )
        embs = codec.embs()
        print(
            f"{mode} mode, the codec has {embs.shape[0]} codebooks, each codebook has {embs.shape[1]} entries, each entry has {embs.shape[2]} dimensions"
        )
        if mode == "decode":
            input = torch.zeros(batch_size, 10, 1).long().to(device)
            with torch.no_grad():
                output = codec(input)
        else:
            with torch.no_grad():
                output = codec(sig)

        if mode == "reconstruct":
            save_dir = os.path.join(root_path, "codecs", "reconstruction_wav")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"wavtokenizer_reconstruction.wav")
            torchaudio.save(save_path,output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0),codec.orig_sample_rate)
            print(f"{mode} mode has been saved to {save_path}")
        elif mode == "encode":
            print(f"{mode} mode, the output shape is {output[0].shape}")
        else:
            print(f"{mode} mode, the output shape is {output.shape}")
