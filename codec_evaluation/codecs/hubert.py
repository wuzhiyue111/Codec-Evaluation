""" Hubert (see https://arxiv.org/abs/2106.07447)."""

import os
import sys
import torch
import codec_evaluation
import json
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec

__all__ = ["Hubert"]

class Hubert(Codec):
    def __init__(
        self,
        sample_rate,
        mode="unquantized_emb",
        need_resample=True,
        model_ckpt_dir=None,
        feature_extractor_config_path=None,
    ):
        """
            sample_rate: sample rate of the input signal
            need_resample: boolean, whether to resample the audio after decoding
            mode: "unquantized_emb"
            model_ckpt_dir: path to the model checkpoint
            feature_extractor_config_path: path to the feature extractor config file
        """
        modes = ["unquantized_emb"]
        if mode not in modes:
            raise ValueError(f"Mode must be one of the following: {modes}, hubert have only support unquantized_emb.")
        try:
            # Workaround to avoid name collisions with installed modules
            root_path = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_path not in x]
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            sys.path = sys_path
        except ImportError:
            raise ImportError("pip install transformers>=4.45.1` to use this module")  
        
        super().__init__(sample_rate, 16000, mode)
        self.need_resample = need_resample
        self.dim = 768     # encoder output dim
        if model_ckpt_dir is None:
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        else:
            self.model = HubertModel.from_pretrained(model_ckpt_dir)

        if feature_extractor_config_path is None:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        else:
            with open(feature_extractor_config_path, "r") as f:
                feature_extractor_config = json.load(f)  
                self.feature_extractor = Wav2Vec2FeatureExtractor(**feature_extractor_config)

        self.hop_length = 320    # downsampling rate of the encoder
        self.token_rate = self.orig_sample_rate / self.hop_length

    @torch.no_grad()
    def embs(self):
        """
            this encoder doesn't have codebooks, raise NotImplementedError
        """
        pass

    # override
    def _sig_to_unquantized_emb(self, sig, length=None):
        """
            sig: [B, T]
            return: [B, D, N]
        """
        features = self.feature_extractor(
            sig,
            sampling_rate=self.orig_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = features["input_values"].squeeze(0).to(device)
        hidden_states = self.model(input_values).last_hidden_state    # [B, N, D]
        unquantized_emb = hidden_states.permute(0, 2, 1)  # [B, D, N]
        return unquantized_emb

    # override
    def _sig_to_quantized_emb(self, sig, length=None):
        """
            not application for this encoder, raise NotImplementedError
        """
        pass

    # override
    def _sig_to_toks(self, sig, length=None):
        """
            not application for this encoder, raise NotImplementedError
        """
        pass

    # override
    def _toks_to_sig(self, toks, length=None):
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
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)  # [B=2, T]
    
    mode = "unquantized_emb"
    codec = (
        Hubert(
            sample_rate,
            mode=mode,
            need_resample=False,    # means the output sample rate is the same as codec's sample rate
            model_ckpt_dir="/sdb/model_weight/codec_evaluation/codec_ckpt/hubert/hubert-base-ls960",
            feature_extractor_config_path="/sdb/model_weight/codec_evaluation/codec_ckpt/hubert/hubert-base-ls960/preprocessor_config.json"
        )
        .eval()
        .to(device)
    )
    with torch.no_grad():
        output = codec(sig)

    print(f"{mode} mode, the output shape is {output.shape}")

