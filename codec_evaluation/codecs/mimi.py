# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Mimi (see https://kyutai.org/Moshi.pdf)."""
import sys
import torch
import codec_evaluation
import os
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec


__all__ = ["Mimi"]


class Mimi(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=8,
        need_resample=True,
    ):
        """
            sample_rate: sample rate of the input signal
            mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            num_codebooks: number of codebooks
            need_resample: Boolean, whether to resample the audio after decoding
        """
        try:
            from transformers import MimiModel
        except ImportError:
            raise ImportError("`pip install transformers>=4.45.1` to use this module")

        super().__init__(sample_rate, 24000, mode)     
        self.num_codebooks = num_codebooks
        self.need_resample = need_resample
        self.vocab_size = 2048
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        self.dim = self.model.config.hidden_size
        
        # Delete the decoder to save memory overhead.
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None
            self.model.decoder_transformer = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.encoder_transformer = None

    # override
    @torch.no_grad()
    def embs(self):
        semantic_layers = self.model.quantizer.semantic_residual_vector_quantizer.layers
        acoustic_layers = self.model.quantizer.acoustic_residual_vector_quantizer.layers
        layers = (semantic_layers + acoustic_layers)[: self.num_codebooks]
        embs = [layer.codebook.embed for layer in layers]
        embs = torch.stack(embs)  # [K, C, H]
        return embs

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
            return: [B, D, N] 
        """
        sig, padding_mask = self.process_sig(sig, length)
        embeddings = self.model.encoder(sig)
        encoder_outputs = self.model.encoder_transformer(embeddings.transpose(1, 2))
        embeddings = encoder_outputs[0].transpose(1, 2)
        unquantized_feats = self.model.downsample(embeddings)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N] 
        """
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, num_quantizers=self.num_codebooks
        )
        toks = output.audio_codes.movedim(-1, -2)  # [B, N, K]
        quantized_feats = self.model.quantizer(
            toks.movedim(-1, -2)    # [B, K, N]
        )
        return quantized_feats
    
    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K]
        """
        sig, padding_mask = self.process_sig(sig, length)
        output = self.model.encode(
            sig, padding_mask, num_quantizers=self.num_codebooks
        )
        toks = output.audio_codes.movedim(-1, -2)  
        return toks

    # override
    def _toks_to_sig(self, toks, length):
        """
            toks: [B, N, K]
            return: [B, T]
        """
        output = self.model.decode(toks.movedim(-1, -2))
        sig = output.audio_values[:, 0]  
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
            Mimi(sample_rate, mode=mode, num_codebooks=num_codebooks).eval().to(device)
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
            save_path = os.path.join(save_dir, f'mimi_reconstruction.wav')
            torchaudio.save(save_path, output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0), codec.orig_sample_rate)
            print(f'{mode} mode has been saved to {save_path}')
        else:
            print(f'{mode} mode, the output shape is {output.shape}') 