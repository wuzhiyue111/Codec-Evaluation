""" YuE (see https://arxiv.org/abs/2503.08638)"""

import os 
import sys
import torch
import codec_evaluation
from omegaconf import OmegaConf
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)

from codec_evaluation.codecs.codec import Codec

all = ["YuE"]

class YuE(Codec):
    def __init__(
        self,
        sample_rate,
        need_resample=True,
        mode="reconstruct",
        num_codebooks=8,
        model_ckpt_dir=None,
    ):
        """
            sample_rate: sample rate of the input signal
            need_resample: boolean, whether to resample the audio after decoding
            mode: "encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"
            num_codebooks: number of codebooks
            model_ckpt_dir: path to the model checkpoint
        """
        from codec_evaluation.codecs.YuE.models.soundstream_hubert_new import SoundStream

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks


        config_path = os.path.join(model_ckpt_dir, 'config.yaml')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"{config_path} file does not exist.")
        config = OmegaConf.load(config_path)
        generator_config = config.generator.config
        self.model = SoundStream(**generator_config)
        model_file = os.path.join(model_ckpt_dir, 'ckpt_00360000.pth')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}.")
        parameter_dict = torch.load(model_file, map_location='cpu', weights_only=False)
        self.model.load_state_dict(parameter_dict['codec_model'])  

        self.vocab_size = 1024
        self.need_resample = need_resample
        self.hop_length = 320
        self.dim = 128
        self.token_rate = self.model.frame_rate

        # Delete the decoder to save memory overhead.
        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            self.model.decoder_2 = None
            self.model.decoder_semantic = None
        elif mode == "decode":
            self.model.encoder = None
            self.model.encoder_semantic = None

    # override
    @torch.no_grad()
    def embs(self):
        # H means the dimension of the embedding
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
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
        _, unquantized_feats = self.model.encode(sig[:, None])
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length):
        """
            sig: [B, T]
            return: [B, D, N]   [2, 1024, 468]
        """
        toks, _ = self.model.encode(sig[:, None])
        toks = toks[: self.num_codebooks]  # [K, B, N]
        quantized_feats = self.model.quantizer.decode(toks)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        """
            sig: [B, T]
            return: [B, N, K]  [2, 468, 8]
        """
        toks, _ = self.model.encode(sig[:, None])  # [K, B, N]
        toks = toks[: self.num_codebooks].movedim(-3, -1)  # [K, B, N]
        return toks, None
        

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
            toks: [B, N, K]
            return: [B, T]   [2, 3200]
        """
        toks = toks.movedim(-1, -3)  # [K, B, N]
        sig = self.model.decode(toks)[:, 0]  # [B, T]
        return sig

if __name__ == "__main__":
    import torchaudio

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2
    num_codebooks = 8

    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "example.wav"))
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1)    # [B=2, T]

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            YuE(
            sample_rate, 
            mode=mode, 
            num_codebooks=num_codebooks,
            model_ckpt_dir="/sdb/model_weight/codec_evaluation/codec_ckpt/yue",
            need_resample=False,
            )
            .eval()
            .to(device)
        )
        embs = codec.embs()
        print(
            f"{mode} mode, the codec has {embs.shape[0]} codebooks, each codebook has {embs.shape[1]} entries, each entry has {embs.shape[2]} dimensions"
        )
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
            save_path = os.path.join(save_dir, f'yue_reconstruction.wav')
            torchaudio.save(
                save_path, 
                output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0),
                codec.orig_sample_rate
            )
            print(f"{mode} mode has been saved to {save_path}")
        elif mode == "encode":
            print(f"{mode} mode, the output shape is {output[0].shape}")
        else:
            print(f"{mode} mode, the output shape is {output.shape}")

