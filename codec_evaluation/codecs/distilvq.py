# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""DistilVQ (Distillation Vector-Quantized Audio Model using Qwen2Audio)."""

import os
import sys
import torch
import gc
import json
import warnings
import torch.nn.functional as F
import numpy as np
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec

# Add Echodec project root to system path first to resolve 'sfm' imports
echodec_root = "/home/lr/project/Echodec"
if (echodec_root not in sys.path):
    sys.path.append(echodec_root)

# Import DistilVQ Models
sys.path.append("/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model")
from distil_vq_model import Qwen2Audio_DistilVQ  # type: ignore
from distil_vq_lit_model import Qwen2Audio_DistilVQ_LitModel  # type: ignore


__all__ = ["DistilVQCodec"]


class DistilVQCodec(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        model_ckpt_path=None,
        need_resample=True,
        orig_sample_rate=16000,
        feature_extractor_config_path="/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json",
    ):
        """
        sample_rate: sample rate of the input signal
        mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            encode: encode the audio to id tokens
            decode: decode the id tokens to audio (not implemented)
            reconstruct: encode -> decode (not fully implemented)
            unquantized_emb: encode -> unquantized embedding
            quantized_emb: encode -> quantized embedding
        model_ckpt_path: path to the model checkpoint
        need_resample: boolean, whether to resample the audio
        orig_sample_rate: original sample rate of the model (default: 16000)
        feature_extractor_config_path: path to the feature extractor config file
        """
        try:
            from transformers import WhisperFeatureExtractor
        except ImportError:
            raise ImportError("`pip install transformers>=4.45.1` to use this module")

        super().__init__(sample_rate, orig_sample_rate, mode)
        self.need_resample = need_resample
        self.max_audio_length_sec = 30  # Maximum audio length in seconds
        
        # Initialize feature extractor for audio preprocessing
        if (feature_extractor_config_path and os.path.exists(feature_extractor_config_path)):
            with open(feature_extractor_config_path, 'r') as f:
                feat_config = json.load(f)
                self.feature_extractor = WhisperFeatureExtractor(**feat_config)
                self.mel_hop_length = feat_config.get("hop_length", 160)  # Default to 160 if not specified
        else:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
            self.mel_hop_length = 160  # Default WhisperFeatureExtractor hop_length

        # Initialize model - load from checkpoint
        if model_ckpt_path:
            if os.path.exists(model_ckpt_path):
                # Load LitModel checkpoint and extract DistilVQ model from it
                try:
                    # First try to load complete LitModel
                    lit_model = Qwen2Audio_DistilVQ_LitModel.load_from_checkpoint(
                        model_ckpt_path,
                        map_location='cpu'
                    )
                    self.model = lit_model.model  # Extract inner DistilVQ model
                    del lit_model  # Delete LitModel to save memory
                except Exception as e:
                    print(f"Failed to load as LitModel, trying direct load: {e}")
                    # If not LitModel, try to load state dict directly
                    checkpoint = torch.load(model_ckpt_path, map_location='cpu')
                    
                    # Try to extract config from checkpoint
                    model_config = None
                    if 'model_config' in checkpoint:
                        model_config = checkpoint['model_config']
                    elif 'hyper_parameters' in checkpoint and 'model_config' in checkpoint['hyper_parameters']:
                        model_config = checkpoint['hyper_parameters']['model_config']
                    
                    if model_config is None:
                        raise ValueError("Cannot find model configuration in checkpoint")
                        
                    # Initialize model with extracted config
                    self.model = Qwen2Audio_DistilVQ(model_config)
                    
                    # Load state dict
                    if 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'])
                    elif 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        # Process Lightning model state_dict, need to remove 'model.' prefix
                        cleaned_state_dict = {}
                        for key, value in checkpoint['state_dict'].items():
                            if key.startswith('model.'):
                                cleaned_state_dict[key[6:]] = value  # Remove 'model.' prefix
                            else:
                                cleaned_state_dict[key] = value
                        self.model.load_state_dict(cleaned_state_dict)
                    else:
                        # Assume direct model state dict
                        self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Model checkpoint not found at {model_ckpt_path}")
        else:
            raise ValueError("model_ckpt_path is required for DistilVQCodec")
        
        # Get model dimensions from config
        config = self.model.config
        self.dim = config.get('vq_dim', 256)  # Default to 256 if not specified
        self.vocab_size = config.get('codebook_size', 8192)  # Default to 8192 if not specified
        self.hop_length = 640  # Derived from feature extractor and model downsampling
        self.token_rate = self.orig_sample_rate / self.hop_length
        
        # Set model to eval mode
        self.model.eval()
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
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
            warnings.warn(f"Audio exceeds maximum length of {self.max_audio_length_sec} seconds. Truncating.")
            abs_lens = torch.clamp(abs_lens, max=max_samples)
        
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
        
        # Check if padding to fixed length is needed
        valid_mel_length = 100 * self.max_audio_length_sec
        batch_size, n_mels, curr_len = input_features.shape
        if curr_len < valid_mel_length:
            padded_features = torch.zeros((batch_size, n_mels, valid_mel_length), 
                                dtype=input_features.dtype, 
                                device=device)
            padded_features[:, :, :curr_len] = input_features
            input_features = padded_features
            del padded_features
        
        # Calculate mel spectrogram attention mask
        raw_music_mask = features.attention_mask.to(device)  # [B, T]
        
        # Check if padding to fixed length is needed for mask
        fixed_length = self.orig_sample_rate * self.max_audio_length_sec
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
        """Return all embedding vectors [K, C, H]"""
        # Get the codebook embeddings from VQ layer
        codes = torch.arange(self.vocab_size, device=next(self.model.parameters()).device)
        embeddings = self.model.vq.get_codes_from_indices(codes.unsqueeze(0)).squeeze(0)
        return embeddings.unsqueeze(0)  # [1, vocab_size, codebook_dim]

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]    [batch_size, distil_output_dim, sequence_length]
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Use model to get unquantized embeddings
        with torch.no_grad():
            # Get features from encoder
            result = self.model.mel_to_embed(
                input_features, 
                attention_mask=attention_mask,
                quantized=False,
                use_distillation=True  # Use distillation path
            )
            
            # Get distilled features from result
            distilled_features = result.get("distilled_features", None)
            if distilled_features is None:
                # Fallback to original features if distilled not available
                distilled_features = result["embeddings"]
        
        # Transpose to expected output format [B, D, N]
        unquantized_emb = distilled_features.transpose(1, 2)
        
        return unquantized_emb

    # override
    def _sig_to_quantized_emb(self, sig, length, use_decoder=False):
        """
            sig: [B, T]
            return: [B, D, N]    [batch_size, vq_output_dim, sequence_length]
            
            use_decoder:
             - True: return final reconstructed features (after MLP upsampler)
             - False: return direct quantized features (after VQ)
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Use model to get quantized embeddings
        with torch.no_grad():
            # Use mel_to_embed with quantized=True and final_output based on use_decoder
            result = self.model.mel_to_embed(
                input_features, 
                attention_mask=attention_mask,
                quantized=True,
                final_output=use_decoder
            )
            
            # Get embeddings from result
            embeddings = result["embeddings"]  # [B, T_enc, D]
        
        # Transpose to expected output format [B, D, N]
        return embeddings.transpose(1, 2)

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K], padding_mask
                   [batch_size, sequence_length, 1], [batch_size, sequence_length]
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Get indices using mel_to_id method
        with torch.no_grad():
            result = self.model.mel_to_id(input_features, attention_mask)
            indices, indices_mask = result["indices"], result["indices_mask"]
        
        # Format indices to match expected output [B, N, K]
        # DistilVQ only has one codebook, so K=1
        toks = indices.unsqueeze(-1)
        
        return toks, indices_mask

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
            toks: [B, N, K]
            return: [B, T]    [batch_size, time_samples]
            
            This method is not implemented for DistilVQCodec as it doesn't support
            direct decoding to waveform.
        """
        raise NotImplementedError("DistilVQCodec doesn't support decoding tokens to audio signals")

    def __call__(self, sig, length, use_decoder=False):
        """
            Override __call__ to support additional parameters
            
            sig: Input signal
            length: Signal length proportions
            use_decoder: For quantized_emb mode only - whether to use transformer decoder
                         True: return features after transformer decoder
                         False: return features before transformer decoder (after VQ)
        """
        if self.mode == "encode":
            return self._sig_to_toks(sig, length)
        elif self.mode == "decode":
            return self._toks_to_sig(sig, length)
        elif self.mode == "reconstruct":
            toks, padding_mask = self._sig_to_toks(sig, length)
            return self._toks_to_sig(toks, length, padding_mask)
        elif self.mode == "unquantized_emb":
            return self._sig_to_unquantized_emb(sig, length)
        elif self.mode == "quantized_emb":
            print(f"Using decoder: {use_decoder}")
            return self._sig_to_quantized_emb(sig, length, use_decoder=use_decoder)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


if __name__ == "__main__":
    import librosa

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    batch_size = 2

    # Test setup
    audio_path = "/path/to/test/audio.mp3"
    if os.path.exists(audio_path):
        sig, sample_rate = librosa.load(audio_path, sr=16000)
        # Cut to 30 seconds of audio, 16kHz sampling rate
        duration = 30 * 16000
        sig = torch.tensor(sig[:duration]).unsqueeze(0).to(device)  # [B=1, T]
        sig = torch.cat([sig, sig], dim=0).to(device)  # [B=2, T]
        length = torch.tensor([0.4, 0.1], device=device)  # Use partial audio

        # Example config path
        feature_config_path = "/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json"
        ckpt_path = "/path/to/distilvq/checkpoint.ckpt"  # Replace with actual path

        # Test all supported modes
        for mode in ["encode", "unquantized_emb", "quantized_emb"]:
            print(f"\nTesting {mode} mode...")
            try:
                codec = DistilVQCodec(
                    sample_rate,
                    mode=mode,
                    model_ckpt_path=ckpt_path,
                    need_resample=True,
                    feature_extractor_config_path=feature_config_path if os.path.exists(feature_config_path) else None
                ).eval().to(device)
                
                with torch.no_grad():
                    output = codec(sig, length)
                    if mode == "encode":
                        toks, mask = output
                        print(f"{mode} mode, the output shape is {toks.shape}, mask shape is {mask.shape}")
                    else:
                        print(f"{mode} mode, the output shape is {output.shape}")
                        print(f"Features statistics: min={output.min().item()}, max={output.max().item()}, mean={output.mean().item()}")

                    # Get embeddings info
                    embs = codec.embs()
                    print(f"The codec has codebook size: {embs.shape[1]}, embedding dimension: {embs.shape[2]}")
            except Exception as e:
                print(f"Error testing {mode} mode: {e}")
    else:
        print(f"Test audio file not found: {audio_path}")