# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Qwen2 Audio Encoder for feature extraction."""

import os
import sys
import torch
import gc
import json
import warnings
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import codec_evaluation
import torchaudio
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec


__all__ = ["Qwen2Encoder"]


class Qwen2Encoder(Codec):
    def __init__(
        self,
        sample_rate,
        mode="unquantized_emb",
        model_ckpt_dir=None,
        need_resample=True,
        orig_sample_rate=16000,
        feature_extractor_config_path="/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json",
    ):
        """
        sample_rate: sample rate of the input signal
        mode: Only "unquantized_emb" is supported for this encoder
        model_ckpt_dir: path to the model checkpoint
        need_resample: boolean, whether to resample the audio
        orig_sample_rate: original sample rate of the model (default: 16000)
        feature_extractor_config_path: path to the feature extractor config file
        """
        try:
            from transformers import WhisperFeatureExtractor, Qwen2AudioConfig, Qwen2AudioForConditionalGeneration
        except ImportError:
            raise ImportError("`pip install transformers>=4.45.1` to use this module")

        super().__init__(sample_rate, orig_sample_rate, mode)
        self.need_resample = need_resample
        self.dim = 1280  # Qwen2Audio encoder output dimension
        self.max_audio_length_sec = 30  # Maximum audio length in seconds
        
        # Initialize feature extractor for audio preprocessing
        # If a config path is provided, load it and initialize feature extractor with those parameters
        if feature_extractor_config_path and os.path.exists(feature_extractor_config_path):
            with open(feature_extractor_config_path, 'r') as f:
                feat_config = json.load(f)
                self.feature_extractor = WhisperFeatureExtractor(**feat_config)
                # Extract hop_length from config if available, otherwise use default
                self.mel_hop_length = feat_config.get("hop_length", 160)  # Default to 160 if not specified
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
            self.mel_hop_length = 160  # Default WhisperFeatureExtractor hop_length

        self.hop_length = 640
        
        # Load model
        if model_ckpt_dir is None:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_ckpt_dir)
            
        # Keep only the encoder part to save memory
        self.encoder = self.model.audio_tower
        self.model = None
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Calculate token rate based on actual hop_length
        self.token_rate = self.orig_sample_rate / self.hop_length

    def process_sig(self, sig, length):
        """
        Process raw audio signal to mel features and attention mask for encoder
        
        Args:
            sig (torch.Tensor): Raw audio signal [B, T]
            length (torch.Tensor): Proportion of the signal to use [B]
            
        Returns:
            tuple: (input_features, attention_mask)
                - input_features: mel features [B, n_mels, T]
                - attention_mask: attention mask for encoder [B, 1, T_enc, T_enc]
        """
        batch_size = sig.shape[0]
        device = sig.device
        max_length = sig.shape[1]
        
        # Calculate absolute lengths for each sample
        abs_lens = (sig.shape[1] * length).long()
        
        # Check if any audio exceeds maximum length
        max_samples = self.max_audio_length_sec * self.orig_sample_rate
        if torch.any(abs_lens > max_samples):
            raise ValueError(f"Audio exceeds maximum length of {self.max_audio_length_sec} seconds")
        
        # Process each audio sample to get features
        raw_audio_list = []
        for i in range(batch_size):
            # Extract valid audio data based on length
            valid_length = min(abs_lens[i].item(), max_length)
            audio_sample = sig[i, :valid_length].cpu().numpy()
            raw_audio_list.append(audio_sample)
            
        # Extract features using Whisper feature extractor
        features = self.feature_extractor(
            raw_audio_list,
            sampling_rate=self.orig_sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move features to the correct device
        input_features = features.input_features.to(device)  # [B, n_mels, T]
        valid_mel_length = 100 * self.max_audio_length_sec
        # Pad input_features to fixed length if needed
        batch_size, n_mels, curr_len = input_features.shape
        if curr_len < valid_mel_length:
            padded_features = torch.zeros((batch_size, n_mels, valid_mel_length), 
                                    dtype=input_features.dtype, 
                                    device=device)
            padded_features[:, :, :curr_len] = input_features
            input_features = padded_features
            del padded_features    
                
        # Calculate mel spectrogram attention mask with downsampling
        # Convert original sampling rate to feature rate
        raw_music_mask = features.attention_mask.to(device)  # [B, T]
        fixed_length = self.orig_sample_rate * self.max_audio_length_sec
        batch_size = raw_music_mask.shape[0]
        if raw_music_mask.shape[1] < fixed_length:
            padded_mask = torch.zeros((batch_size, fixed_length), 
                                    dtype=raw_music_mask.dtype, 
                                    device=device)
            padded_mask[:, :raw_music_mask.shape[1]] = raw_music_mask
            raw_music_mask = padded_mask
            del padded_mask

        feature_attention_mask = F.max_pool1d(
            raw_music_mask.unsqueeze(1).float(),
            kernel_size=self.mel_hop_length,
            stride=self.mel_hop_length
        )
        feature_attention_mask = (feature_attention_mask.squeeze(1) > 0)
        
        # Calculate encoder output attention mask dimensions
        batch_size, n_mels, seq_len = input_features.shape
        feature_lengths = feature_attention_mask.sum(-1)  # [B]
        encoder_lengths = ((feature_lengths - 2) // 2 + 1).long()  # [B] accounting for downsampling
        
        # Create full attention mask for encoder
        max_len = ((seq_len - 2) // 2 + 1)
        attention_mask = torch.ones((batch_size, 1, max_len, max_len), device=device)
        
        # Apply masking for each sample
        for i, length in enumerate(encoder_lengths):
            length_val = length.item() if torch.is_tensor(length) else int(length)
            attention_mask[i, :, :, length_val:] = float('-inf')
            attention_mask[i, :, length_val:, :] = float('-inf')
            
        return input_features, attention_mask

    # override
    @torch.no_grad()
    def embs(self):
        # This codec doesn't have discrete codebooks, so we return None
        return None

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]    [batch_size, 1280, sequence_length]
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Get hidden states from Qwen2 encoder
        with torch.no_grad():
            encoder_output = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask
            )
        
        hidden_states = encoder_output.last_hidden_state  # [B, N, D]
        
        # Calculate valid mask for output
        if attention_mask is not None:
            valid_mask = attention_mask[:, 0, 0, :].float()
            # Handle encoder downsampling
            if valid_mask.size(1) != hidden_states.size(1):
                valid_mask = F.max_pool1d(valid_mask.unsqueeze(1), kernel_size=2, stride=2).squeeze(1) > 0
            hidden_mask = valid_mask
        else:
            batch_size, seq_length, _ = hidden_states.shape
            hidden_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
        
        # Transpose to match the expected output format [B, D, N]
        unquantized_emb = hidden_states.transpose(1, 2)
        
        return unquantized_emb

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
        Not applicable for this encoder, but return the same as unquantized
        """
        warnings.warn(
            "Qwen2Encoder does not support true quantization. "
            "Returning unquantized embeddings instead. "
            "Be careful when using this output as it's not actually quantized.",
            UserWarning
        )
        return self._sig_to_unquantized_emb(sig, length)

    # override
    def _sig_to_toks(self, sig, length):
        """
        Not applicable for this encoder, but provide a compatible implementation
        """
        # Process audio to get unquantized embeddings
        unquantized_emb = self._sig_to_unquantized_emb(sig, length)
        batch_size, _, seq_len = unquantized_emb.shape
        
        # Create padding mask based on length
        input_features, attention_mask = self.process_sig(sig, length)
        valid_mask = attention_mask[:, 0, 0, :].float()
        if valid_mask.size(1) != seq_len:
            valid_mask = F.max_pool1d(valid_mask.unsqueeze(1), kernel_size=2, stride=2).squeeze(1) > 0
        
        # Return dummy tokens and the padding mask
        dummy_tokens = torch.zeros(batch_size, seq_len, 1, device=sig.device, dtype=torch.long)
        return dummy_tokens, valid_mask

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
        Not applicable for this encoder, raise NotImplementedError
        """
        raise NotImplementedError("Qwen2Encoder does not support decoding tokens to signal")


if __name__ == "__main__":
    import librosa

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2

    audio_path = "/sdb/data1/music/mix_music/Fine-Grained-Music/processed_data/audio/MUSIC_000001.mp3"
    sig, sample_rate = librosa.load(audio_path, sr=16000)
    # chunk the audio in 30s, 16k sample rate
    duration = 30 * sample_rate
    sig = torch.tensor(sig[:duration]).unsqueeze(0).to(device)  # [B=1, T]
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)  # [B=2, T]

    length = torch.tensor([0.4, 0.1], device=device)

    # Example path to feature extractor config
    feature_config_path = "/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json"

    mode = "unquantized_emb"  # Only mode supported
    codec = (
        Qwen2Encoder(
            sample_rate,
            mode=mode,
            model_ckpt_dir=None,
            need_resample=True,
            feature_extractor_config_path=feature_config_path if os.path.exists(feature_config_path) else None
        )
        .eval()
        .to(device)
    )
    
    with torch.no_grad():
        output = codec(sig, length)
        
    print(f"{mode} mode, the output shape is {output.shape}")
    print(f"Features statistics: min={output.min().item()}, max={output.max().item()}, mean={output.mean().item()}")