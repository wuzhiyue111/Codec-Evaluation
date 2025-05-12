# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""SemantiCodec (see https://arxiv.org/abs/2405.00233)."""

import os
import sys
import torch
import codec_evaluation

root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec


__all__ = ["SemantiCodec"]


_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


class SemantiCodec(Codec):
    TOKEN_RATES = [25, 50, 100]
    SEMANTIC_VOCAB_SIZES = [4096, 8192, 16384, 32768]

    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        token_rate=100,
        semantic_vocab_size=8192,
        ddim_sample_step=50,
        cfg_scale=2.0,
        model_ckpt_dir: str | None = None,
        need_resample=True,
    ):
        """ "
        sample_rate: sample rate of the input signal
        mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            encode: encode the audio to id tokens
            decode: decode the id tokens to audio
            reconstruct: encode -> decode
            unquantized_emb: encode -> unquantized embedding
            quantized_emb: encode + quantizer -> quantized embedding
        token_rate: token rate of the codec
        semantic_vocab_size: semantic vocab size of the codec
        ddim_sample_step: number of steps for DDIM sampling
        cfg_scale: classifier free guidance scale
        model_ckpt_dir: path to the model checkpoint
        need_resample: boolean, whether to resample the audio after decoding
        """
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]

            global semanticodec
            import semanticodec

            sys.path = sys_path
        except ImportError:
            raise ImportError(
                "`pip install git+https://github.com/haoheliu/SemantiCodec-inference.git` to use this module"
            )
        super().__init__(sample_rate, 16000, mode)
        self.token_rate = token_rate
        self.semantic_vocab_size = semantic_vocab_size
        self.cfg_scale = cfg_scale
        self.need_resample = need_resample
        self.num_codebooks = 2
        self.acoustic_vocab_size = 8192
        self.model = semanticodec.SemantiCodec(
            token_rate=token_rate,
            semantic_vocab_size=semantic_vocab_size,
            ddim_sample_step=ddim_sample_step,
            cfg_scale=cfg_scale,
            checkpoint_path=model_ckpt_dir,
            cache_path=_CACHE_DIR,
        ).to("cpu")
        self.hop_length = int(self.orig_sample_rate * 2 / self.token_rate)
        self.dim = self.model.encoder.feature_dimension
        self.vocab_size = 8192

        # Delete the decoder to save memory overhead.
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None

    # override
    @torch.no_grad()
    def to(self, *args, **kwargs):
        self.model.encoder.centroid_npy = self.model.encoder.centroid_npy.to(
            *args, **kwargs
        )
        return super().to(*args, **kwargs)

    # override
    @torch.no_grad()
    def embs(self):
        # H means the dimension of the embedding
        if self.semantic_vocab_size != 8192:
            raise NotImplementedError("The size of acoustic codebook is fixed to 8192")
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.semantic_vocab_size, device=device)
        toks = (
            toks[:, None, None].expand(-1, -1, self.num_codebooks).clone()
        )  # [C, 1, K]
        embs = self._token_to_quantized_feature(toks)
        embs = torch.cat(
            embs.split(embs.shape[-1] // self.num_codebooks, dim=-1), dim=-2
        )  # [C, K, H]
        embs = embs.movedim(0, 1)  # [K, C, H]
        return embs

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        """
        sig: [B, T]
        return: [B, D, C, N]  C: token type(acoustic and semantic)  [2, 768, 2, 472]
        """
        toks, _ = self._sig_to_toks(sig, length)
        unquantized_feats = self.model.encoder.unquant(toks)
        unquantized_feats = unquantized_feats.permute(0, 3, 2, 1)  
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
        sig: [B, T]
        return: [B, D, N]  D: cat acoustic_feature and semantic_feature dim  [2, 1536, 472]  
        """
        toks, _ = self._sig_to_toks(sig, length)
        quantized_feats = self._token_to_quantized_feature(toks)
        quantized_feats = quantized_feats.movedim(1, 2)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
        sig: [B, T]
        return: [B, N, K]   [2, 472, 2]
        """
        toks = self._encode(sig)
        return toks, None

    # override

    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
        toks: [B, N, K]
        return: [B, T]  [2, 3200]
        """
        sig = self._decode(toks)[:, 0]
        return sig

    # See https://github.com/haoheliu/SemantiCodec-inference/blob/8dc464c3385d2389a695ed3f718f4a0caf3ed33f/semanticodec/main.py
    def _token_to_quantized_feature(self, tokens):
        semantic_tokens, acoustic_tokens = tokens[..., 0], tokens[..., 1]
        semantic_feature = self.model.encoder.unquant(semantic_tokens)
        token_num, feature_dim = semantic_feature.shape[-2], semantic_feature.shape[-1]
        acoustic_feature = self.model.encoder.quantizer.get_output_from_indices(
            acoustic_tokens
        ).reshape(-1, token_num, feature_dim)
        return torch.cat([acoustic_feature, semantic_feature], dim=-1)

    def _encode(self, waveform):
        # Calculate the original duration
        original_duration = waveform.shape[1] / semanticodec.main.SAMPLE_RATE
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
            semanticodec.main.AUDIOMAE_PATCH_DURATION
            - original_duration % semanticodec.main.AUDIOMAE_PATCH_DURATION
        )
        # Calculate the token length in theory
        target_token_len = (
            8
            * original_duration
            / semanticodec.main.AUDIOMAE_PATCH_DURATION
            / self.model.stack_factor_K
        )
        segment_sample_length = int(
            semanticodec.main.SAMPLE_RATE * semanticodec.main.SEGMENT_DURATION
        )
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations
        if waveform.shape[1] % segment_sample_length < segment_sample_length:
            diff = int(
                segment_sample_length - waveform.shape[1] % segment_sample_length
            )
            waveform = torch.nn.functional.pad(waveform, [0, diff])

        mel_target_length = semanticodec.main.MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        # Calculate the mel spectrogram
        mels = [
            semanticodec.main.extract_kaldi_fbank_feature(
                x[None], semanticodec.main.SAMPLE_RATE, target_length=mel_target_length
            )["ta_kaldi_fbank"]
            for x in waveform
        ]
        mel = torch.stack(mels)
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0
        tokens = self.model.encoder(mel.to(waveform.device))
        tokens = tokens[:, : semanticodec.main.math.ceil(target_token_len), :]
        return tokens

    def _decode(self, tokens):
        windowed_token_list = self.model.encoder.long_token_split_window(
            tokens,
            window_length=int(512 / self.model.stack_factor_K),
            overlap=semanticodec.main.SEGMENT_OVERLAP_RATIO,
        )
        windowed_waveform = []
        for _, windowed_token in enumerate(windowed_token_list):
            latent = self._token_to_quantized_feature(windowed_token)
            latent = torch.cat(
                [
                    latent,
                    torch.ones(
                        latent.shape[0],
                        int(512 / self.model.stack_factor_K) - latent.shape[1],
                        latent.shape[2],
                    ).to(latent.device)
                    * -1,
                ],
                dim=1,
            )
            waveform = self.model.decoder.generate_sample(
                latent,
                ddim_steps=self.model.ddim_sample_step,
                unconditional_guidance_scale=self.model.cfg_scale,
            )
            windowed_waveform.append(waveform)
        output = semanticodec.main.overlap_add_waveform(
            windowed_waveform,
            overlap_duration=semanticodec.main.SEGMENT_DURATION
            * semanticodec.main.SEGMENT_OVERLAP_RATIO,
        )
        # Each patch step equal 16 mel time frames, which have 0.01 second
        trim_duration = (tokens.shape[1] / 8) * 16 * 0.01 * self.model.stack_factor_K
        return torch.as_tensor(
            output[..., : int(trim_duration * semanticodec.main.SAMPLE_RATE)],
            device=tokens.device,
        )


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
            SemantiCodec(
                sample_rate,
                mode=mode,
                model_ckpt_dir='/sdb/model_weight/codec_evaluation/codec_ckpt/semantic',
                need_resample=False
            )
            .eval()
            .to(device)
        )
        embs = codec.embs()
        print(f'{mode} mode, the codec has {embs.shape[0]} codebooks, each codebook has {embs.shape[1]} entries, each entry has {embs.shape[2]} dimensions')
        if mode == "decode":
            input = torch.zeros(batch_size, 10, 2).long().to(device)
            with torch.no_grad():
                output = codec(input)
        else:
            with torch.no_grad():
                output = codec(sig)

        if mode == "reconstruct":
            save_dir = os.path.join(root_path, "codecs", "reconstruction_wav")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'semanticodec_reconstruction.wav')
            torchaudio.save(save_path, output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0), codec.orig_sample_rate)
            print(f'{mode} mode has been saved to {save_path}')
        elif mode == "encode":
            print(f'{mode} mode, the output shape is {output[0].shape}')
        else:
            print(f'{mode} mode, the output shape is {output.shape}')