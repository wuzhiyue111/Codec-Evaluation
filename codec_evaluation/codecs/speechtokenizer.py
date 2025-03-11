# ==============================================================================
# Copyright 2024 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""SpeechTokenizer (see https://arxiv.org/abs/2308.16692)."""

import os
import sys
import torch
import torch.nn.functional as F
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)


from huggingface_hub import snapshot_download
from codec_evaluation.codecs.codec import Codec


__all__ = ["SpeechTokenizer"]


class SpeechTokenizer(Codec):
    def __init__(
        self,
        sample_rate,
        mode="reconstruct",
        num_codebooks=8,
        need_resample=True,
        model_ckpt_dir=None,
    ):
        """
            sample_rate: sample rate of the input signal
            mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            num_codebooks: number of codebooks
            need_resample: boolean, whether to resample the audio after decoding
            model_ckpt_dir: path to the model checkpoint
        """
        try:
            # Workaround to avoid name collisions with installed modules
            root_dir = os.path.dirname(os.path.realpath(__file__))
            sys_path = [x for x in sys.path]
            sys.path = [x for x in sys.path if root_dir not in x]
            import speechtokenizer

            sys.path = sys_path
        except ImportError:
            raise ImportError("`pip install speechtokenizer` to use this module")

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks
        if model_ckpt_dir is None:
            source = "fnlp/SpeechTokenizer"
            path = snapshot_download(repo_id=source)
        else:
            path = model_ckpt_dir
        config_path = os.path.join(path, "speechtokenizer_hubert_avg", "config.json")
        checkpoint_path = os.path.join(
            path, "speechtokenizer_hubert_avg", "SpeechTokenizer.pt"
        )
        self.model = speechtokenizer.SpeechTokenizer.load_from_checkpoint(
            config_path, checkpoint_path
        )
        self.need_resample = need_resample
        self.hop_length = self.model.encoder.hop_length
        self.dim = self.model.encoder.dimension
        self.token_rate = self.model.sample_rate / self.model.downsample_rate

        # Delete the decoder to save memory overhead.
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.transform = None

    # override
    @torch.no_grad()
    def embs(self):
        # H means the dimension of the embedding
        # See https://github.com/ZhangXInFD/SpeechTokenizer/blob/a9f88dc72642b600654a62861e34342babae6c71/speechtokenizer/quantization/core_vq.py#L360
        vocab_size = 1024
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(vocab_size, device=device)
        toks = (
            toks[None, :, None].expand(self.num_codebooks, -1, -1).clone()
        )  # [K, C, 1]
        embs = []
        for i, indices in enumerate(toks):
            layer = self.model.quantizer.vq.layers[i]
            quantized = layer.decode(indices)  # [C, H, 1]
            embs.append(quantized)
        assert (self.model.quantizer.decode(toks) == sum(embs)).all()
        embs = torch.stack(embs)[..., 0]  # [K, C, H] 
        return embs

    # override
    def _sig_to_unquantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]   [2, 1024, 468]
        """
        if sig.dim() == 2:
            sig = sig.unsqueeze(1)
        unquantized_feats = self.model.encoder(sig)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]   [2, 1024, 468]
        """
        toks = self.model.encode(sig[:, None])[: self.num_codebooks]  # [K, B, N]    
        quantized_feats = self.model.quantizer.decode(toks)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K]   [2, 468, 8]
        """
        toks = self.model.encode(sig[:, None])[: self.num_codebooks]  # [K, B, N]
        toks = toks.movedim(-3, -1)  
        return toks, None

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
            toks: [B, N, K]
            return: [B, T]   [2, 3200]
        """
        toks = toks.movedim(-1, -3)  # [K, B, N]
        sig = self.model.decode(toks)[:, 0]  
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
            SpeechTokenizer(
                sample_rate,
                mode=mode,
                num_codebooks=num_codebooks,
                model_ckpt_dir='/sdb/model_weight/codec_evaluation/codec_ckpt/speechtokenizer',
                need_resample=False,
            )
            .eval()
            .to(device)
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
            save_path = os.path.join(save_dir, f'speechtokenizer_reconstruction.wav')
            torchaudio.save(save_path, output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0), codec.orig_sample_rate)
            print(f'{mode} mode has been saved to {save_path}')
        elif mode == "encode":
            print(f'{mode} mode, the output shape is {output[0].shape}')
        else:
            print(f'{mode} mode, the output shape is {output.shape}')