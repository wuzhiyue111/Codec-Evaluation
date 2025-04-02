"""X-codec (see https://arxiv.org/pdf/2408.17175)."""

import os
import sys
import torch
import torchaudio.transforms as T
import codec_evaluation
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)


from codec_evaluation.codecs.codec import Codec

__all__ = ["XCodec"]

class XCodec(Codec):
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
            encode: encode the audio to id tokens
            decode: decode the id tokens to audio
            reconstruct: encode -> decode
            unquantized_emb: encode -> unquantized embedding
            quantized_emb: encode + quantizer -> quantized embedding
        model_ckpt_dir: path to the model checkpoint
        """
        # Workaround to avoid name collisions with installed modules
        root_dir = os.path.dirname(os.path.realpath(__file__))
        sys_path = [x for x in sys.path]
        sys.path = [x for x in sys.path if root_dir not in x]
        from .xcodec import SoundStream

        sys.path = sys_path

        super().__init__(sample_rate, 16000, mode)
        self.num_codebooks = num_codebooks
        self.model = SoundStream
        self.vocab_size = 1024
        self.need_resample = need_resample
        self.hop_length = self.model.hop_length
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
        # See https://github.com/zhenye234/xcodec/blob/main/quantization/core_vq.py#L356
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
    """
        TODO: Tomorrow I will need to debug every piece of code and conduct tests
    """
    def _sig_to_unquantized_emb(self, sig, length):
        unquantized_feats = self.model.encoder(sig)
        return unquantized_feats
    
    # override
    def _sig_to_quantized_emb(self, sig, length):
        toks = self.model.encode(sig)
        quantized_feats = self.model.quantizer.decode(toks)
        return quantized_feats

    # override
    def _sig_to_toks(self, sig, length):
        toks = self.model.encode(sig)  
        toks = toks.movedim(-3, -1)
        return toks, None  

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        toks = toks.movedim(-1, -3)
        sig = self.model.decode(toks)
        return sig


if __name__ == "__main__":
    import torchaudio
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 2
    num_codebooks = 8
    model_ckpt_dir= ''
    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "example.wav"))
    sig = sig.unsqueeze(0)
    sig = torch.cat([sig, sig], dim=0).to(device).squeeze(1) # [B=2, T]

    for mode in ["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"]:
        codec = (
            XCodec(
                sample_rate, 
                mode=mode,
                num_codebooks=num_codebooks,
                need_resample=False,
                model_ckpt_dir=model_ckpt_dir,
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
            save_path = os.path.join(save_dir, f'xcodec_reconstruction.wav')
            torchaudio.save(save_path, output[0].unsqueeze(0).cpu() if use_cuda else output[0].unsqueeze(0), codec.orig_sample_rate)
            print(f'{mode} mode has been saved to {save_path}')
        elif mode == "encode":
            print(f'{mode} mode, the output shape is {output[0].shape}')
        else:
            print(f'{mode} mode, the output shape is {output.shape}')