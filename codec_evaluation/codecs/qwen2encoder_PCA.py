# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Qwen2 Audio Encoder with PCA for feature extraction and dimensionality reduction."""

import os
import sys
import torch
import gc
import json
import warnings
import torch.nn.functional as F
import numpy as np
import pickle
import codec_evaluation
from sklearn.decomposition import PCA as SklearnPCA

root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)
from codec_evaluation.pca.incremental_pca import IncrementalPCA
from codec_evaluation.codecs.qwen2encoder import Qwen2Encoder


__all__ = ["Qwen2EncoderPCA"]


class Qwen2EncoderPCA(Qwen2Encoder):
    def __init__(
        self,
        sample_rate,
        mode="unquantized_emb",
        model_ckpt_dir=None,
        need_resample=True,
        orig_sample_rate=16000,
        feature_extractor_config_path="/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json",
        target_dim=256,
        pca_model_path=None,
    ):
        """
        sample_rate: sample rate of the input signal
        mode: Only "unquantized_emb" is supported for this encoder
        model_ckpt_dir: path to the model checkpoint
        need_resample: boolean, whether to resample the audio
        orig_sample_rate: original sample rate of the model (default: 16000)
        feature_extractor_config_path: path to the feature extractor config file
        target_dim: target dimension after PCA reduction (default: 256)
        pca_model_path: path to pre-trained PCA model (.pth file for torch model or .pkl file for sklearn)
        """
        super().__init__(
            sample_rate, 
            mode,
            model_ckpt_dir,
            need_resample,
            orig_sample_rate,
            feature_extractor_config_path
        )
        
        # Original dimension from Qwen2 encoder
        self.source_dim = 1280
        # Target dimension after PCA
        self.target_dim = target_dim
        # Update the dimension attribute to reflect PCA output
        self.dim = self.target_dim
        
        # Initialize PCA
        self.pca = None
        self.pca_initialized = False
        self.is_torch_pca = False
        
        # Load pre-trained PCA model if provided
        if pca_model_path and os.path.exists(pca_model_path):
            self.load_pca_model(pca_model_path)
            print(f"Loaded PCA model from {pca_model_path}")
    
    def load_pca_model(self, model_path):
        """Load a pre-trained PCA model
        
        Args:
            model_path (str): Path to the model file (.pth for torch model, .pkl for sklearn)
        """
        try:
            # Check file extension to determine loading method
            if model_path.endswith('.pth'):
                # Load torch IncrementalPCA model
                self.pca = IncrementalPCA(n_components=self.target_dim, n_features=self.source_dim)
                self.pca.load_state_dict(torch.load(model_path))
                self.pca.eval()
                self.is_torch_pca = True
                print(f"Pre-trained torch PCA model loaded from {model_path}")
                
            elif model_path.endswith('.pkl'):
                # Load sklearn PCA model
                with open(model_path, 'rb') as f:
                    self.pca = pickle.load(f)
                self.is_torch_pca = False
                print(f"Pre-trained sklearn PCA model loaded from {model_path}")
                if hasattr(self.pca, 'explained_variance_ratio_'):
                    print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
            else:
                raise ValueError(f"Unsupported model file extension: {model_path}. Use .pth for torch or .pkl for sklearn models.")
            
            self.pca_initialized = True
            
            # Verify the PCA output dimension
            if hasattr(self.pca, 'n_components'):
                actual_components = self.pca.n_components
            else:
                actual_components = self.pca.components_.shape[0]
                
            if actual_components != self.target_dim:
                warnings.warn(
                    f"The loaded PCA model has {actual_components} components, "
                    f"but target_dim is {self.target_dim}. Using the loaded model's dimension."
                )
                self.target_dim = actual_components
                self.dim = self.target_dim
                
        except Exception as e:
            warnings.warn(f"Failed to load PCA model from {model_path}: {e}")
            self.pca_initialized = False
    
    def save_pca_model(self, model_path):
        """Save the current PCA model
        
        Args:
            model_path (str): Path to save the PCA model (.pth for torch, .pkl for sklearn)
        """
        if not self.pca_initialized:
            warnings.warn("PCA model is not initialized, nothing to save.")
            return
            
        try:
            if self.is_torch_pca:
                torch.save(self.pca.state_dict(), model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.pca, f)
            print(f"PCA model saved to {model_path}")
        except Exception as e:
            warnings.warn(f"Failed to save PCA model to {model_path}: {e}")
    
    def _initialize_pca(self, data):
        """Initialize PCA from data if no pre-trained model was loaded
        
        Args:
            data (torch.Tensor): Data of shape [B, D, N]
        """
        # Reshape data to 2D for PCA
        batch_size, dim, seq_len = data.shape
        data_2d = data.permute(0, 2, 1).reshape(-1, dim).cpu().numpy()
        
        # Initialize and fit PCA (removed random_state parameter)
        self.pca = SklearnPCA(n_components=self.target_dim)
        self.pca.fit(data_2d)
        self.is_torch_pca = False
        self.pca_initialized = True
        print(f"PCA initialized with explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
    
    def _apply_pca(self, data):
        """Apply PCA to reduce dimensionality
        
        Args:
            data (torch.Tensor): Data of shape [B, D, N]
            
        Returns:
            torch.Tensor: Reduced data of shape [B, target_dim, N]
        """
        device = data.device
        batch_size, dim, seq_len = data.shape
        
        # Initialize PCA if not already done
        if not self.pca_initialized:
            self._initialize_pca(data)
            
        if self.is_torch_pca:
            # For torch IncrementalPCA model
            # Reshape to the format expected by our IncrementalPCA
            data_reshaped = data.permute(0, 2, 1).reshape(-1, dim)
            
            # Apply transformation using torch model
            with torch.no_grad():
                reduced_data_2d = self.pca.transform(data_reshaped)
            
            # Reshape back to original format with reduced dimension
            reduced_data = reduced_data_2d.reshape(batch_size, seq_len, self.target_dim).permute(0, 2, 1)
        else:
            # For sklearn PCA model
            # Reshape to 2D for PCA transformation
            data_2d = data.permute(0, 2, 1).reshape(-1, dim).cpu().numpy()
            
            # Apply transformation using sklearn model
            reduced_data_2d = self.pca.transform(data_2d)
            
            # Reshape back to original format with reduced dimension
            reduced_data = torch.from_numpy(reduced_data_2d).float().to(device)
            reduced_data = reduced_data.reshape(batch_size, seq_len, self.target_dim).permute(0, 2, 1)
        
        return reduced_data

    # Override parent method
    def _sig_to_unquantized_emb(self, sig, length):
        """
        Apply Qwen2 encoder and then reduce dimensionality with PCA
        
        Args:
            sig (torch.Tensor): Raw audio signal [B, T]
            length (torch.Tensor): Proportion of the signal to use [B]
            
        Returns:
            torch.Tensor: Reduced dimensional embeddings [B, target_dim, N]
        """
        # Get original embeddings from parent class
        unquantized_emb = super()._sig_to_unquantized_emb(sig, length)
        
        # Apply PCA for dimensionality reduction
        reduced_emb = self._apply_pca(unquantized_emb)
        
        return reduced_emb

    # Override if needed
    def _sig_to_quantized_emb(self, sig, length):
        """
        Not applicable for this encoder, but return the same as unquantized with PCA
        """
        warnings.warn(
            "Qwen2EncoderPCA does not support true quantization. "
            "Returning PCA-reduced unquantized embeddings instead. "
            "Be careful when using this output as it's not actually quantized.",
            UserWarning
        )
        return self._sig_to_unquantized_emb(sig, length)


if __name__ == "__main__":
    import librosa

    use_cuda = torch.cuda.is_available()
    device = "cuda:1" if use_cuda else "cpu"
    batch_size = 2
    sample_rate = 16000
    target_dim = 256

    audio_path = "/sdb/data1/music/mix_music/Fine-Grained-Music/processed_data/audio/MUSIC_000001.mp3"
    sig, sample_rate = librosa.load(audio_path, sr=16000)
    # chunk the audio in 30s, 16k sample rate
    duration = 30 * sample_rate
    sig = torch.tensor(sig[:duration]).unsqueeze(0).to(device)  # [B=1, T]
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)  # [B=2, T]

    length = torch.tensor([0.4, 0.1], device=device)

    # Example path to feature extractor config
    feature_config_path = "/home/lr/project/Echodec/sfm/sfm_model/whisperVQ_model/whisper_feature_config.json"
    
    # Path to pre-trained PCA model
    pca_model_path = "/home/lr/project/Codec-Evaluation/codec_evaluation/pca/pca_model_1280to256.pth"

    mode = "unquantized_emb"  # Only mode supported
    codec = (
        Qwen2EncoderPCA(
            sample_rate,
            mode=mode,
            model_ckpt_dir=None,
            need_resample=True,
            target_dim=target_dim,
            feature_extractor_config_path=feature_config_path if os.path.exists(feature_config_path) else None,
            pca_model_path=pca_model_path if os.path.exists(pca_model_path) else None
        )
        .eval()
        .to(device)
    )
    
    with torch.no_grad():
        output = codec(sig, length)
        
    print(f"{mode} mode, the output shape is {output.shape}")
    print(f"Features statistics: min={output.min().item()}, max={output.max().item()}, mean={output.mean().item()}")
