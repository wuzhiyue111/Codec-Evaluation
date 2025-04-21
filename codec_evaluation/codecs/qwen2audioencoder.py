"""qwen2_audio encoder (see https://arxiv.org/pdf/2407.10759)."""

import os
import sys
import torch
import json
import torch.nn.functional as F
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec

__all__ = ["Qwen2AudioEncoder"]


class Qwen2AudioEncoder(Codec):
    def __init__(
        self,
        sample_rate,
        need_resample=True,
        mode="unquantized_emb",
        model_ckpt_dir=None,
        feature_extractor_config_path=None,
    ):
        """
            sample_rate: sample rate of the input signal
            need_resample: boolean, whether to resample the audio after decoding
            mode: "unquantized_emb"
            model_ckpt_dir: path to the model checkpoint
            feature_extractor_config_path: path to the feature extractor config
        """
        modes = ["unquantized_emb"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of the following: {modes}, qwen2audioencoder have only support unquantized_emb.")
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            from transformers import WhisperFeatureExtractor, Qwen2AudioForConditionalGeneration

            sys.path = sys_path
        except ImportError:
            raise ImportError("pip install transformers>=4.45.1` to use this module")  
        
        super().__init__(sample_rate, 16000, mode)
        self.need_resample = need_resample
        self.dim = 1280     # encoder output dim
        self.audio_max_length_sec = 30  # max audio length in seconds
        if model_ckpt_dir is None:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_ckpt_dir)
        
        if feature_extractor_config_path is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
            self.mel_hop_length = 160   # Default WhisperFeatureExtractor hop_length
        else :
            with open(feature_extractor_config_path, "r") as f:
                feature_extractor_config = json.load(f)
                self.feature_extractor = WhisperFeatureExtractor(**feature_extractor_config)
                self.mel_hop_length = feature_extractor_config.get("hop_length", 160)

        """
            why self.hop_length = self.mel_hop_length * 4 ?
            16000 * 30ms = 480000   480000/160 = 3000   3000 / 4 = 750(2 conv1D(2 stride) + 1 avgpool1D(2 stride))
            so encoder's hop_length = 160 * (3000/750) = 640
        """
        self.hop_length = self.mel_hop_length * 4   
        self.token_rate = self.orig_sample_rate / self.hop_length
        # Keep only the encoder part to save memory
        self.encoder = self.model.audio_tower
        self.model = None   

    def process_sig(self, sig, length):
        """ 
            Process the signal to obtain input_feature and feature_attention_mask
            sig: [B, T]
            length: [B]
            return: tuple(input_feature, feature_attention_mask)
                    input_feature: [B, D, N]
                    feature_attention_mask: [B, 1, N_enc, N_enc]
        """
        device = sig.device
        batch_size = sig.shape[0] 
        max_length = sig.shape[1]

        # Calculate absolute lengths for each sample
        abs_lens = (sig.shape[1] * length).long()   # tensor

        # Check if any audio exceeds maximum length
        max_samples = self.audio_max_length_sec * self.orig_sample_rate
        if torch.any(abs_lens > max_samples):
            raise ValueError(f"Audio length exceeds the maximum allowed length of {self.audio_max_length_sec} seconds.")
        
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
        
        """
            Convert the maximum duration of the audio (in seconds) 
            to the number of time steps of the Mel-spectrogram.
            mel_hop_length = 160, so the number of time steps of the Mel-spectrogram is:
            1/ (160 / orig_sample_rate) = 1 / 0.01 = 100 
        """
        # Ensure that all batches of `input_features` have the same, predefined `valid_mel_length` length, even if the original audio is very short. 
        input_features = features.input_features.to(device)     # [B, n_mels, T]
        valid_mel_length = self.audio_max_length_sec * 100
        batch_size, n_mels, max_mel_seq_len = input_features.shape
        if max_mel_seq_len < valid_mel_length:
            pad_features = torch.zeros((batch_size, n_mels, valid_mel_length),
                                      dtype=input_features.dtype, 
                                      device=device)
            pad_features[:, :, :max_mel_seq_len] = input_features
            input_features = pad_features
            del pad_features
        
        # Calculate mel spectrogram attention mask with downsampling
        # Convert original sampling rate to feature rate, ensure that the original audio mask also corresponds to the unified maximum length. 
        raw_audio_mask = features.attention_mask.to(device)   # [B, T]
        fixed_length = self.audio_max_length_sec * self.orig_sample_rate
        batch_size = raw_audio_mask.shape[0]
        if raw_audio_mask.shape[1] < fixed_length:
            pad_mask = torch.zeros((batch_size, fixed_length),
        if raw_audio_mask.shape[1] < fixed_length:
            pad_mask = torch.zeros((batch_size, fixed_length),
                                   dtype=raw_audio_mask.dtype,
                                   device=device)
            pad_mask[:, :raw_audio_mask.shape[1]] = raw_audio_mask
            raw_audio_mask = pad_mask
            del pad_mask
        
        # downsample the attention mask `raw_audio_mask` of the original audio to the temporal dimension of the Mel spectrogram. 
        feature_attention_mask = F.max_pool1d(
            raw_audio_mask.unsqueeze(1).float(),    # shape [B, 1, T]
            kernel_size=self.mel_hop_length,        # pooling kernel size
            stride=self.mel_hop_length,             # pooling stride
        ).squeeze(1)

        # boolean mask
        feature_attention_mask = feature_attention_mask > 0

        # Calculate encoder output attention mask dimensions
        batch_size, n_mels, max_mel_seq_len = input_features.shape
        feature_lengths = feature_attention_mask.sum(dim=-1)    # [B]
        encoder_output_lengths = ((feature_lengths - 2) // 2 + 1).long()    # [B] accounting for downsampling

        # Create full attention mask for encoder
        max_length = ((max_mel_seq_len - 2) // 2 + 1)
        attention_mask = torch.ones((batch_size, 1, max_length, max_length), device = device)
        
        # Apply masking for each sample
        for i, len in enumerate(encoder_output_lengths):
            len_val = len.item() if torch.is_tensor(len) else int(len)
            attention_mask[i, :, len_val:, :] = float('-inf')
            attention_mask[i, :, :, len_val:] = float('-inf')

        return input_features, attention_mask

    @torch.no_grad()
    def embs(self):
        """
            this encoder doesn't have codebooks,raise NotImplementedError
        """
        pass

    #override
    def _sig_to_unquantized_emb(self, sig, length):
        """ 
            
            sig: [B, T]
            return: [B, D, N]
        """
        input_features, feature_attention_mask = self.process_sig(sig, length)

        with torch.no_grad():
            encoder_output = self.encoder(
                input_features,
                attention_mask=feature_attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_output.last_hidden_state    # [B, N, D]
        unquantized_emb = hidden_states.permute(0, 2, 1)  # [B, D, N]
        return unquantized_emb

    #override
    def _sig_to_quantized_emb(self, sig, length):
        """
            not application for this encoder, raise NotImplementedError 
        """
        pass

    #override
    def _sig_to_toks(self, sig, length, padding_mask=None):
        """
            not application for this encoder, raise NotImplementedError
        """
        pass

    #override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
            not application for this encoder, raise NotImplementedError 
        """
        pass

if __name__ == "__main__":
    import torchaudio
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2

    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "example.wav"))
    sig = sig[:sample_rate * 30].clone().detach().unsqueeze(0).to(device) # [B=1, T]
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1) # [B=2, T]
    # length = torch.tensor([1.0, 0.5]).to(device) # [B=2]
    
    mode = "unquantized_emb"
    codec = (
        Qwen2AudioEncoder(
            sample_rate, 
            mode=mode,
            need_resample=False, # means the output sample rate is the same as codec's sample rate
            model_ckpt_dir = "/mnt/sda/a6000/sdb/data1/model_weight/codec_evaluation/codec_ckpt/qwen2audioencoder",
            feature_extractor_config_path = "/mnt/sda/a6000/sdb/model_weight/whisper-large-v3/preprocessor_config.json"
        )
        .eval()
        .to(device)
    )
    with torch.no_grad():
        output = codec(sig)
    
    print(f"{mode} mode, the output shape is {output.shape}")
        
    
    

