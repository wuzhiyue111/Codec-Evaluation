# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""QwenAudioVQ (Vector-Quantized Audio Model using Qwen2Audio)."""

import os
import sys
import torch
import gc
import json
import warnings
import torch.nn.functional as F
import numpy as np
import torchaudio
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec

# Add Echodec project root to system path first to resolve 'sfm' imports
echodec_root = "/home/lr/project/Echodec"
if echodec_root not in sys.path:
    sys.path.append(echodec_root)

# Import QwenAudioVQModel and LitModel
sys.path.append("/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model")
from vq_model import QwenAudioVQModel # type: ignore
from vq_lit_model import QwenAudioVQLitModel # type: ignore


__all__ = ["QwenAudioVQ"]


class QwenAudioVQ(Codec):
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
        model_ckpt_path: path to the model checkpoint (LitModel checkpoint)
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
                # Load LitModel checkpoint and extract QwenAudioVQModel from it
                try:
                    # First try to load complete LitModel
                    lit_model = QwenAudioVQLitModel.load_from_checkpoint(
                        model_ckpt_path,
                        map_location='cpu'
                    )
                    self.model = lit_model.model  # Extract inner VQ model
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
                    self.model = QwenAudioVQModel(model_config)
                    
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
            raise ValueError("model_ckpt_path is required for QwenAudioVQ")
        
        qwen2audio = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        
        # Ensure pretrained encoder is correctly copied, not just referenced
        if hasattr(qwen2audio, 'audio_tower') and qwen2audio.audio_tower is not None:
            # If the model has state_dict method, use deep copy to ensure weights are correctly copied
            if hasattr(self.model, 'pretrained_encoder') and self.model.pretrained_encoder is not None:
                # If there is already a pretrained encoder, load new weights
                self.model.pretrained_encoder.load_state_dict(qwen2audio.audio_tower.state_dict())
            else:
                # If there is no pretrained encoder, directly copy the entire module
                self.model.pretrained_encoder = type(qwen2audio.audio_tower)(**qwen2audio.audio_tower.config.to_dict())
                self.model.pretrained_encoder.load_state_dict(qwen2audio.audio_tower.state_dict())
            
            print("Successfully loaded Qwen2Audio pretrained encoder weights")
        else:
            warnings.warn("Could not find audio_tower component in Qwen2Audio model")

        del qwen2audio
        
        # Get model dimensions from config
        config = self.model.config
        self.dim = config.get('vq_dim', 256)  # Default to 8 if not specified
        self.vocab_size = config.get('codebook_size', 8192)  # Default to 8192 if not specified
        self.hop_length = 640
        self.token_rate = self.orig_sample_rate / self.hop_length
        
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
        
        # Similar to Qwen2Encoder, check if padding to fixed length is needed
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
        
        # Similar to Qwen2Encoder, check if padding to fixed length is needed
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
        codes = torch.arange(self.vocab_size, device=next(self.model.parameters()).device)
        embeddings = self.model.vq.get_codes_from_indices(codes.unsqueeze(0)).squeeze(0)
        return embeddings.unsqueeze(0)  # [1, vocab_size, codebook_dim]

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]    [batch_size, codebook_dim, sequence_length]
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Use model's encode method to get indices and mask
        with torch.no_grad():
            encoded = self.model.encode(input_features, attention_mask)
            indices, indices_mask = encoded["indices"], encoded["indices_mask"]
            
            # Get features before VQ
            # Here we could use VQ layer's get_codes_from_indices to get unquantized embeddings
            # But this isn't truly "unquantized embedding", it's the corresponding vector in the codebook
            # To get the true unquantized embedding, we need to process the input similar to _sig_to_toks
            
            # Use original implementation to get unquantized embedding
            if hasattr(self.model, 'pretrained_encoder') and self.model.pretrained_encoder is not None:
                audio_outputs = self.model.pretrained_encoder(input_features, attention_mask=attention_mask)
                hidden_states = audio_outputs.last_hidden_state
                
                # Get mask
                if attention_mask is not None:
                    valid_mask = attention_mask[:, 0, 0, :].float()
                    if valid_mask.size(1) != hidden_states.size(1):
                        valid_mask = F.max_pool1d(valid_mask.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
                    hidden_mask = valid_mask > 0
                else:
                    batch_size, seq_length, _ = hidden_states.shape
                    hidden_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
                
                # Process features
                downsampled_features = self.model.mlp_downsampler(hidden_states)
                
                if hasattr(self.model, 'transformer_feature_encoder') and self.model.transformer_feature_encoder is not None:
                    vq_features = self.model.transformer_feature_encoder(downsampled_features, mask=hidden_mask)
                elif hasattr(self.model, 'pre_vq_proj'):
                    vq_features = self.model.pre_vq_proj(downsampled_features)
                else:
                    vq_features = downsampled_features
                '''
                test for feature before transformer
                '''
                vq_features = downsampled_features

            else:
                raise ValueError("Pretrained encoder is required for encoding mode")
        
        # Transpose to expected output format [B, D, N]
        unquantized_emb = vq_features.transpose(1, 2)
        
        return unquantized_emb

    # override
    def _sig_to_quantized_emb(self, sig, length, use_decoder=False):
        """
            sig: [B, T]
            return: [B, D, N]    [batch_size, codebook_dim, sequence_length]
            
            use_decoder:
             - True: return final reconstructed features (after transformer_feature_decoder and mlp_upsampler)
             - False: return direct quantized features (after VQ)
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Use model's encode method to get indices and mask
        with torch.no_grad():
            encoded = self.model.encode(input_features, attention_mask)
            indices, indices_mask = encoded["indices"], encoded["indices_mask"]
            
            # Get quantized embeddings
            quantized = self.model.vq.get_output_from_indices(indices)
            
            # Determine whether to use decoder based on use_decoder parameter
            if use_decoder:
                # Decoder processing flow: vq -> transformer_feature_decoder -> mlp_upsampler
                if not self.model.use_new_structure:
                    if self.model.transformer_feature_decoder is not None:
                        # First through transformer feature decoder
                        decoded_features = self.model.transformer_feature_decoder(quantized, mask=indices_mask)
                    else:
                        decoded_features = quantized
                    
                    # Then through MLP upsampler
                    quantized_emb = self.model.mlp_upsampler(decoded_features)
                else:
                    # Decoder processing with new structure
                    # First through MLP, then through transformer_feature_decoder
                    if hasattr(self.model, 'mlp_upsampler') and self.model.mlp_upsampler is not None:
                        # First through MLP processing
                        processed_features = self.model.mlp_upsampler(quantized)
                    else:
                        processed_features = quantized
                    
                    # Then through transformer feature decoder
                    if hasattr(self.model, 'transformer_feature_decoder') and self.model.transformer_feature_decoder is not None:
                        decoded_features = self.model.transformer_feature_decoder(processed_features, mask=indices_mask)
                    else:
                        decoded_features = processed_features
                    
                    
                    if hasattr(self.model, 'reconstructed_layer_norm') and self.model.reconstructed_layer_norm is not None:
                        quantized_emb = self.model.reconstructed_layer_norm(decoded_features)
                    else:
                        quantized_emb = decoded_features
            else:
                # Without using decoder, directly return quantized embeddings
                quantized_emb = quantized
        
        # Transpose to expected output format [B, D, N]
        quantized_emb = quantized_emb.transpose(1, 2)
        
        return quantized_emb

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K], padding_mask
                   [batch_size, sequence_length, 1], [batch_size, sequence_length]
        """
        # Process audio to mel features and get attention mask
        input_features, attention_mask = self.process_sig(sig, length)
        
        # Directly use model's encode method
        with torch.no_grad():
            encoded = self.model.encode(input_features, attention_mask)
            indices, indices_mask = encoded["indices"], encoded["indices_mask"]
        
        # Format indices to match expected output [B, N, K]
        # QwenAudioVQ only has one codebook, so K=1
        toks = indices.unsqueeze(-1)
        
        return toks, indices_mask

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
            toks: [B, N, K]
            return: [B, T]    [batch_size, time_samples]
            
            This method is not implemented for QwenAudioVQ, as it does not support direct decoding to waveform.
        """
        raise NotImplementedError("QwenAudioVQ does not support decoding tokens to audio signal")


    def __call__(self, sig, length):
        """
            Override __call__ to support additional parameters
            
            sig: Input signal
            length: Signal length proportions
            use_decoder: For quantized_emb mode only - whether to use transformer decoder
                         True: return features after transformer decoder and mlp upsampler
                         False: return features directly after VQ (before decoder)
        """
        if self.mode == "encode":
            return self.sig_to_toks(sig, length)
        elif self.mode == "decode":
            return self.toks_to_sig(sig, length)
        elif self.mode == "reconstruct":
            toks, padding_mask = self._sig_to_toks(sig, length)
            return self.toks_to_sig(toks, length, padding_mask)
        elif self.mode == "unquantized_emb":
            return self.sig_to_unquantized_emb(sig, length)
        elif self.mode == "quantized_emb":
            return self.sig_to_quantized_emb(sig, length)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


if __name__ == "__main__":
    import librosa

    use_cuda = torch.cuda.is_available()
    device = "cuda:1" if use_cuda else "cpu"
    batch_size = 2

    # Following Qwen2Encoder's testing approach
    audio_path = "/sdb/data1/music/mix_music/Fine-Grained-Music/processed_data/audio/MUSIC_000001.mp3"
    sig, sample_rate = librosa.load(audio_path, sr=16000)
    # Cut to 30 seconds of audio, 16kHz sampling rate
    duration = 30 * 16000
    sig = torch.tensor(sig[:duration]).unsqueeze(0).to(device)  # [B=1, T]
    sig = torch.cat([sig, sig], dim=0).to(device)  # [B=2, T]

    length = torch.tensor([0.4, 0.1], device=device)  # Use partial audio, like Qwen2Encoder

    # Example config path
    feature_config_path = "/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json"

    ckpt_path = "/home/lr/project/Echodec/sfm/train/whisperVQ_train/experiments/Qwen2VQ_quantized_256_size_8192/checkpoints/last-v1.ckpt"

    # Test all supported modes
    for mode in ["encode", "unquantized_emb", "quantized_emb"]:
        print(f"\nTesting {mode} mode...")
        codec = (
            QwenAudioVQ(
                sample_rate,
                mode=mode,
                model_ckpt_path=ckpt_path,  # Replace with actual model path when using
                need_resample=True,
                feature_extractor_config_path=feature_config_path if os.path.exists(feature_config_path) else None
            )
            .eval()
            .to(device)
        )
        
        # Get embeddings info
        embs = codec.embs()
        print(f"The codec has codebook size: {embs.shape[1]}, embedding dimension: {embs.shape[2]}")
        
        # Process audio
        with torch.no_grad():
            if mode == "encode":
                output, mask = codec(sig, length)
                print(f"{mode} mode, the output shape is {output.shape}")
            elif mode == "quantized_emb":
                # Test both with and without decoder
                for use_decoder in [False, True]:
                    print(f"  - With use_decoder={use_decoder}:")
                    output = codec(sig, length, use_decoder=use_decoder)
                    print(f"    Output shape: {output.shape}")
                    print(f"    Features statistics: min={output.min().item()}, max={output.max().item()}, mean={output.mean().item()}")
            else:
                output = codec(sig, length)  # Note the length parameter here
                print(f"{mode} mode, the output shape is {output.shape}")
                print(f"Features statistics: min={output.min().item()}, max={output.max().item()}, mean={output.mean().item()}")
