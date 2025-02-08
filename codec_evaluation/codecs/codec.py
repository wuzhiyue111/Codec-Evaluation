# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Codec interface."""

# abc.ABC/abc.abstractmethod，用于定义抽象基类的模块。帮助创建必须被子类实现的方法
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

    def __init__(self, sample_rate, orig_sample_rate, mode="reconstruct"):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"`mode` ({mode}) must be one of {self._MODES}")
        self.sample_rate = sample_rate
        self.orig_sample_rate = orig_sample_rate
        self.mode = mode

    def forward(self, input, length=None):
        # id是否合理？
        # backgorund：tts -》id
        if self.mode == "encode":
            toks = self.sig_to_toks(input, length)
            return toks
        if self.mode == "decode":
            sig = self.toks_to_sig(input, length)
            return sig
        if self.mode == "reconstruct":
            toks = self.sig_to_toks(input, length)
            sig = self.toks_to_sig(toks, length)
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
        # sig: [B, T]
        sig, length = self.process_audio(sig, length)
        return self._sig_to_unquantized_emb(sig, length)
    
    def sig_to_quantized_emb(self, sig, length=None):
        # sig: [B, T]
        sig, length = self.process_audio(sig, length)
        return self._sig_to_quantized_emb(sig, length)

    def sig_to_toks(self, sig, length=None):
        # sig: [B, T]
        sig, length = self.process_audio(sig, length)
        return self._sig_to_toks(sig, length)

    def toks_to_sig(self, toks, length=None):
        # toks: [B, N, K]
        if length is None:
            length = torch.ones(len(toks), device=toks.device)
        sig = self._toks_to_sig(toks, length)
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
