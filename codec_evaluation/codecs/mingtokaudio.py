# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""MingTokAudio (see http://github.com/yongjie-lv/MingTok-Audio)."""

import os
import sys
import torch
import torchaudio
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec


__all__ = ["MingTokAudio"]

class MingTokAudio(Codec):
    def __init__(self,
        sample_rate,
        mode="reconstruct",
        need_resample=True,
        model_ckpt_dir=None
    ):
        """
        sample_rate: sample rate of the input signal
        mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            reconstruct: encode -> decode
            unquantized_emb: encode -> unquantized embedding
        need_resample: boolean, whether to resample the audio after decoding
        model_ckpt_dir: path to the model checkpoint
        """
        modes = ["reconstruct", "unquantized_emb"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of the following: {modes}, mingtokaudio have only support reconstruct and unquantized_emb.")
        # Workaround to avoid name collisions with installed modules
        root_dir = os.path.dirname(os.path.realpath(__file__))
        sys_path = [x for x in sys.path]
        sys.path = [x for x in sys.path if root_dir not in x]
        from codec_evaluation.codecs.MingTok_Audio.audio_tokenizer.modeling_audio_vae import AudioVAE

        sys.path = sys_path

        super().__init__(sample_rate, 16000, mode)
        self.need_resample = need_resample
        self.model_ckpt_dir = model_ckpt_dir

        if model_ckpt_dir is not None:
            self.model = AudioVAE.from_pretrained(model_ckpt_dir)
        else:
            self.model = AudioVAE.from_pretrained("inclusionAI/MingTok-Audio")

        self.hop_length = self.model.config.enc_kwargs.get("hop_size", 320)
        self.token_rate = self.orig_sample_rate / self.hop_length
        if self.model.config.semantic_module_kwargs is not None:
            self.dim = self.model.semantic_emb_dim
        else:
            self.dim = self.model.config.enc_kwargs["latent_dim"]

    def _get_waveform_length(self, sig, length=None):
        if length is None:
            waveform_length = torch.full(
                (sig.size(0),), 
                sig.shape[-1], 
                dtype=torch.long, 
                device=sig.device
            )
        else:
            if length.dtype.is_floating_point:
                waveform_length = (length * sig.shape[-1]).long()
            else:   
                waveform_length = length.long()
        return waveform_length
    
    def forward(self, sig, length=None, padding_mask=None):
        if self.mode == "reconstruct":
            sig, length = self.process_audio(sig, length)
            waveform_length = self._get_waveform_length(sig, length)

            latent, _ = self.model.encode_latent(sig, waveform_length)
            rec_sig = self.model.decode(latent)

            if self.need_resample:
                rec_sig = torchaudio.functional.resample(
                    rec_sig, 
                    self.orig_sample_rate, 
                    self.sample_rate
                )
            return rec_sig
    
        if self.mode == "unquantized_emb":
            return self.sig_to_unquantized_emb(sig, length)

    @torch.no_grad()
    def embs(self):
        raise NotImplementedError("MingTokAudio has no discrete codebook embeddings.")

    def _sig_to_unquantized_emb(self, sig, length):
        waveform_length = self._get_waveform_length(sig, length)
        unified_emb, _, _ = self.model.encode_unified_emb_from_waveform(sig, waveform_length)
        unquantized_feats = unified_emb.permute(0, 2, 1)  # [B, D, T]
        return unquantized_feats
    
    def _sig_to_quantized_emb(self, sig, length):
        raise NotImplementedError("MingTokAudio does not support quantized_emb because it has no quantizer/codebook.")

    def _sig_to_toks(self, sig, length):
        raise NotImplementedError("MingTokAudio does not support encode mode because it has no discrete token ids.")

    def _toks_to_sig(self, toks, length, padding_mask=None):
        raise NotImplementedError("MingTokAudio does not support decode mode from discrete token ids.")


if __name__ == "__main__":
    import torchaudio

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2

    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "music.mp3"))
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)  # [B=2, T]
    # if sig.shape[0] > 1:
    #     sig = sig.mean(dim=0, keepdim=True)  # Convert to mono if stereo
    # sig = sig.to(device)

    for mode in ["reconstruct", "unquantized_emb"]:
        codec = (
            MingTokAudio(
                sample_rate, 
                mode=mode, 
                model_ckpt_dir="/sdb/model_weight/codec_evaluation/codec_ckpt/mingtok_audio",
                need_resample=False
            )
            .eval()
            .to(device)
        )
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = codec(sig)
            else:
                output = codec(sig)

        if mode == "reconstruct":
            save_dir = os.path.join(root_path, "codecs", "reconstruction_wav")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"mingtokaudio_reconstruction.wav")
            torchaudio.save(
                save_path,
                output[0].cpu() if use_cuda else output[0],
                codec.orig_sample_rate,
            )
            print(f"{mode} mode has been saved to {save_path}")
        else:
            print(f"{mode} mode, the output shape is {output.shape}")

