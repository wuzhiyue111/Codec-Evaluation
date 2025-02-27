# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Codec interface."""

from abc import ABC, abstractmethod

import torch
import torchaudio


__all__ = ["Codec"]


# B: batch size
# T: sequence length in the time domain
# N: sequeunce length in the token domain
# C: vocabulary size (assuming that each codebook has the same number of tokens)
# K: number of codebooks
class Codec(torch.nn.Module, ABC):
    _MODES = ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]

    def __init__(self, sample_rate, orig_sample_rate, mode="reconstruct", need_resample=True):
        """
            sample_rate: sample rate of the input signal
            orig_sample_rate: original sample rate of the codec
            mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
                encode: encode the audio to id tokens
                decode: decode the id tokens to audio
                reconstruct: encode -> decode
                unquantized_emb: encode -> unquantized embedding
                quantized_emb: encode + quantizer -> quantized embedding
            need_resample: Boolean, mode == 'reconstruct' or 'deocde' default True, whether to resample the audio after decoding
        """
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"`mode` ({mode}) must be one of {self._MODES}")
        self.sample_rate = sample_rate
        self.orig_sample_rate = orig_sample_rate
        self.mode = mode
        self.need_resample = need_resample

    def forward(self, input, length=None):
        if self.mode == "encode":
            toks, padding_mask = self.sig_to_toks(input, length)
            return toks, padding_mask
        if self.mode == "decode":
            sig = self.toks_to_sig(input, length)
            return sig
        if self.mode == "reconstruct":
            toks, padding_mask = self.sig_to_toks(input, length)
            sig = self.toks_to_sig(toks, length, padding_mask)
            return sig
        if self.mode == "unquantized_emb":
            unquantized_emb = self.sig_to_unquantized_emb(input, length)
            return unquantized_emb
        if self.mode == "quantized_emb":
            quantized_emb = self.sig_to_quantized_emb(input, length)
            return quantized_emb
        
    def process_audio(self, sig, length=None):
        # sig: [B, T]
        sig = torchaudio.functional.resample(
            sig,
            self.sample_rate,
            self.orig_sample_rate,
        )
        if length is None:
            length = torch.ones(len(sig), device=sig.device)
        return sig, length

    def sig_to_unquantized_emb(self, sig, length=None):
        # sig:[B, T]
        sig, length = self.process_audio(sig, length)
        return self._sig_to_unquantized_emb(sig, length)
    
    def sig_to_quantized_emb(self, sig, length=None):
        # sig:[B, T]
        sig, length = self.process_audio(sig, length)
        return self._sig_to_quantized_emb(sig, length)

    def sig_to_toks(self, sig, length=None):
        # sig:[B, T]
        sig, length = self.process_audio(sig, length)
        return self._sig_to_toks(sig, length)

    def toks_to_sig(self, toks, length=None, padding_mask=None):
        # toks: [B, N, K]
        if length is None:
            length = torch.ones(len(toks), device=toks.device)
        sig = self._toks_to_sig(toks, length, padding_mask)
        if self.need_resample:
            sig = torchaudio.functional.resample(
                sig,
                self.orig_sample_rate,
                self.sample_rate,
            )
        return sig

    @abstractmethod
    def embs(self):
        raise NotImplementedError

    @abstractmethod
    def _sig_to_toks(self, sig, length):
        # sig: [B, T]
        raise NotImplementedError

    @abstractmethod
    def _toks_to_sig(self, toks, length):
        # toks: [B, N, K]
        raise NotImplementedError
    
    @abstractmethod
    def _sig_to_unquantized_emb(self, sig, length):
        # sig: [B, T]
        raise NotImplementedError
    
    @abstractmethod
    def _sig_to_quantized_emb(self, sig, length):
        # sig: [B, T]
        raise NotImplementedError
