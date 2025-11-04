""" Mucodec from levo (see https://github.com/tencent-ailab/songgeneration)
"""

import os 
import sys
import torch
import codec_evaluation
from codec_evaluation.codecs.codec import Codec
root_path = codec_evaluation.__path__[0]
from codec_evaluation.utils.seperator import Separator
from codec_evaluation.codecs.levo_modules.audio_tokenizer import Flow1dVAESeparate

all = ["MuCodec"]

class MuCodec(Codec):
    def __init__(
        self,
        sample_rate: int,
        mode="reconstruct",
        num_codebooks=1,
        encoder_ckpt_path=None,
        content_vec_ckpt_path=None,
        model_ckpt_dir=None,
        vae_model_path=None,
        vae_config_path=None,
        separator_kwargs: dict | None = None,
        need_resample=True,
    ):
        # Workaround to avoid name collisions with installed modules
        root_dir = root_path
        sys.path = [x for x in sys.path if root_dir not in x]
           
        from codec_evaluation.codecs.levo_modules.audio_tokenizer import AudioTokenizer
        """Instantiate a compression model."""
        # https://github.com/tencent-ailab/SongGeneration/blob/b0d71f33609530309711feb7ebd7e990b49138f1/codeclm/models/builders.py#L31

        super().__init__(sample_rate, 48000, mode)

        model: Flow1dVAESeparate = AudioTokenizer.get_pretrained(
            model_type = "Flow1dVAESeparate",
            name = model_ckpt_dir, 
            encoder_ckpt_path = encoder_ckpt_path,
            content_vec_ckpt_path = content_vec_ckpt_path,
            vae_config = vae_config_path, 
            vae_model = vae_model_path, 
            mode="extract",
        )

        self.model = model
        self.need_resample = need_resample
        self.hop_length = int(960 * 2) # levo 在模型内做了 resample，因此需要乘 2
        self.dim = 1024     # codec_dim
        self.token_rate = self.orig_sample_rate / self.hop_length
        self.vocab_size = self.model.model.model.rvq_bestrq_bgm_emb.quantizers[0].codebook.weight.shape[0]
        # self.num_codebooks = [16384, 16384]
        self.num_codebooks = num_codebooks
        self.separator = Separator(**(separator_kwargs or {})).eval()
        for param in self.separator.parameters():
            param.requires_grad = False

        if mode == "encode" or mode == "unquantized_emb" or mode == "quantized_emb":
            # Tango
            self.model.model.vae = None
            self.model.model.cfm_wrapper = None
        elif mode == "decode":
            self.model.model.bestrq = None
            self.model.model.rsq48tobestrq = None


    # override
    @torch.no_grad()
    def embs(self):
        # H means the dimension of the embedding
        # See https://github.com/zhenye234/xcodec/blob/main/quantization/core_vq.py#L356
        device = next(iter(self.model.state_dict().values())).device
        toks = torch.arange(self.vocab_size, device=device)
        # toks = (toks[None, :, None].expand(self.num_codebooks, -1, -1).clone())  # [K, C, 1]
        
        # [B x N x T]
        toks = toks[None, None,:, ]
        rvq_bestrq_vocal_emb = self.model.model.model.rvq_bestrq_emb
        vocal_embs, _, _ = rvq_bestrq_vocal_emb.from_codes(toks)
        rvq_bestrq_bgm_emb = self.model.model.model.rvq_bestrq_bgm_emb
        bgm_embs, _, _  = rvq_bestrq_bgm_emb.from_codes(toks)

        return vocal_embs, bgm_embs

    # override
    def _sig_to_unquantized_emb(self, sig, length = None):
        """
            sig: [B, T]
            return: [B, D, N]   [2, 1024, 468]
        """

        # sig, vocal, instr [1, 4329323]
        _, input_audios_vocal, input_audios_bgm = self.separator.separate_from_mix(sig, sample_rate=self.orig_sample_rate)
        
        input_audio_vocal_0 = input_audios_vocal[:,0,:].unsqueeze(1)
        input_audio_vocal_1 = input_audios_vocal[:,1,:].unsqueeze(1)
        input_audio_vocal_0 = self.model.model.preprocess_audio(input_audio_vocal_0)
        input_audio_vocal_1 = self.model.model.preprocess_audio(input_audio_vocal_1)

        input_audio_bgm_0 = input_audios_bgm[:,0,:].unsqueeze(1)
        input_audio_bgm_1 = input_audios_bgm[:,1,:].unsqueeze(1)
        input_audio_bgm_0 = self.model.model.preprocess_audio(input_audio_bgm_0)
        input_audio_bgm_1 = self.model.model.preprocess_audio(input_audio_bgm_1)

        bestrq_emb = self.model.model.model.extract_bestrq_embeds(input_audio_vocal_0,input_audio_vocal_1,self.model.model.layer_vocal)
        bestrq_emb_bgm = self.model.model.model.extract_bestrq_embeds(input_audio_bgm_0,input_audio_bgm_1,self.model.model.layer_bgm)

        unquantized_feats = torch.cat([bestrq_emb, bestrq_emb_bgm], dim=1)
        return unquantized_feats

    # override
    def _sig_to_quantized_emb(self, sig, length = None):
        # sig, vocal, instr [1, 4329323]
        _, input_audios_vocal, input_audios_bgm = self.separator.separate_from_mix(sig, sample_rate=self.orig_sample_rate)

        input_audio_vocal_0 = input_audios_vocal[:,0,:].unsqueeze(1)
        input_audio_vocal_1 = input_audios_vocal[:,1,:].unsqueeze(1)
        input_audio_vocal_0 = self.model.model.preprocess_audio(input_audio_vocal_0)
        input_audio_vocal_1 = self.model.model.preprocess_audio(input_audio_vocal_1)

        input_audio_bgm_0 = input_audios_bgm[:,0,:].unsqueeze(1)
        input_audio_bgm_1 = input_audios_bgm[:,1,:].unsqueeze(1)
        input_audio_bgm_0 = self.model.model.preprocess_audio(input_audio_bgm_0)
        input_audio_bgm_1 = self.model.model.preprocess_audio(input_audio_bgm_1)

        bestrq_emb = self.model.model.model.extract_bestrq_embeds(input_audio_vocal_0,input_audio_vocal_1,self.model.model.layer_vocal)
        bestrq_emb_bgm = self.model.model.model.extract_bestrq_embeds(input_audio_bgm_0,input_audio_bgm_1,self.model.model.layer_bgm)
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.model.model.model.rvq_bestrq_emb(bestrq_emb) # b,d,t

        quantized_bestrq_emb_bgm, codes_bestrq_emb_bgm, *_ =  self.model.model.model.rvq_bestrq_bgm_emb(bestrq_emb_bgm) # b,d,t
        quantized_feats = torch.cat([quantized_bestrq_emb, quantized_bestrq_emb_bgm], dim=1)
        return quantized_feats

    # override
    def _sig_to_toks(self, audio, length): 
        # audio [B,2,T]
        B = audio.shape[0]
        _, vocal_audio, instr_audio = self.separator.separate_from_mix(audio, sample_rate=self.orig_sample_rate)

        [vocal_code, bgm_code], [vocal_bestrq_emb, bgm_bestrq_emb], _ = self.model.model.model.fetch_codes_batch(
            (vocal_audio), (instr_audio), additional_feats=[],
            layer_vocal=self.model.model.layer_vocal, layer_bgm=self.model.model.layer_bgm
        )

        # ids [B, N, K]
        vocal_id = vocal_code.transpose(1, 2)
        instr_id = bgm_code.transpose(1, 2)
        concat_toks = torch.cat([vocal_id, instr_id], dim=-1)
        # valid mask [B, N]
        if length is not None:
            length = length.to(vocal_audio.device)
            # length_v = length[:B]
            # length_i = length[B:]
            length_v = length
            length_i = length
            if not torch.equal(length_v, length_i):
                max_len = max(int(length_v.max().item()), int(length_i.max().item()))
            else:
                max_len = int(length_v.max().item())
            audio_mask = torch.arange(max_len, device=vocal_audio.device).unsqueeze(0).expand(B, -1) < length_v.unsqueeze(1)
            token_length = vocal_id.shape[1]
            if audio_mask.shape[1] != token_length:
                vm = audio_mask.unsqueeze(1).float()
                vm = torch.nn.functional.interpolate(vm, size=token_length, mode='nearest')
                valid_mask = vm.squeeze(1)
            else:
                valid_mask = audio_mask.float()
        else:
            valid_mask = torch.ones(vocal_id.shape[0], vocal_id.shape[1], device=vocal_id.device)
        return concat_toks, valid_mask

    # override
    def _toks_to_sig(self, toks, length, padding_mask=None):
        """
            toks: [B, N, 2]
            return: [B, T]   [2, 3200]
        """
        vocal_toks = toks[:,: , [0]]
        bgm_toks = toks[:,: , [1]]
        vocal_toks = vocal_toks.transpose(-1, -2)
        bgm_toks = bgm_toks.transpose(-1, -2)
        # vocal_toks,bgm_toks = toks.unbind(dim=-1)
        codes = (vocal_toks, bgm_toks)
        sig = self.model.model.code2sound(
            codes, 
            prompt_vocal=None, 
            prompt_bgm=None, 
            guidance_scale=1.5, 
            num_steps=50, 
            disable_progress=False, 
            chunked=True, 
            chunk_size=128
        ) # [B,N,T] -> [B,T]
        return sig


if __name__ == "__main__":
    import torchaudio
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    # diffusion olny support bs=1, for bs > 1,using cycle to reconstrcut every
    batch_size = 1
    num_codebooks = 1

    sig, sample_rate = torchaudio.load(os.path.join(root_path, "codecs", "music.mp3"))
    # sig = sig[:,:sample_rate*10]
    if sig.shape[0] == 1:
        sig = torch.cat([sig, sig], dim=0)
    sig = sig.unsqueeze(0).to(device) # [B,2, T]
    '''download demucs ckpt from
        https://huggingface.co/tencent/SongGeneration/blob/main/third_party/demucs/ckpt/htdemucs.pth
        https://huggingface.co/tencent/SongGeneration/blob/main/third_party/demucs/ckpt/htdemucs.yaml
    '''
    separator_cfg = {
        "dm_model_path": "/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/demucs/htdemucs.pth",
        "dm_config_path": "/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/demucs/htdemucs.yaml",
    }

    '''pretrained assets from the SongGeneration (Levo)
        encoder_ckpt_path https://huggingface.co/tencent/SongGeneration/blob/aa9d1b3b9d9db27012e8250d75feaab3473be370/ckpt/encode-s12k.pt
        content_vec_ckpt_path https://huggingface.co/tencent/SongGeneration/tree/aa9d1b3b9d9db27012e8250d75feaab3473be370/ckpt/models--lengyue233--content-vec-best/snapshots/c0b9ba13db21beaa4053faae94c102ebe326fd68
        model_ckpt_path https://huggingface.co/tencent/SongGeneration/blob/main/ckpt/model_septoken/model_2.safetensors
        vae_model_path https://huggingface.co/tencent/SongGeneration/blob/main/ckpt/vae/autoencoder_music_1320k.ckpt
        vae_config_path https://huggingface.co/tencent/SongGeneration/blob/main/ckpt/vae/stable_audio_1920_vae.json
        
    '''
 
    for mode in ["reconstruct", "decode","encode",  "unquantized_emb", "quantized_emb"]:
        codec = (
            MuCodec(
                sample_rate = sample_rate, 
                mode=mode,
                num_codebooks=num_codebooks,
                separator_kwargs=separator_cfg,
                encoder_ckpt_path = "/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/encode-s12k.pt",
                content_vec_ckpt_path ="/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/models--lengyue233--content-vec-best/snapshots/c0b9ba13db21beaa4053faae94c102ebe326fd68",
                model_ckpt_dir="/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/Flow1dVAESeparate/model_2.safetensors",
                vae_model_path="/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/vae/autoencoder_music_1320k.ckpt",
                vae_config_path = "/sdb/model_weight/codec_evaluation/codec_ckpt/mucodec/vae/stable_audio_1920_vae.json",
                need_resample=False,
            )
            .eval()
            .to(device)
        )
        # embs = codec.embs()
        vocal_embs,bgm_embs = codec.embs()
        print(
            f"{mode} mode, the codec has {vocal_embs.shape[0]} codebooks, each codebook has {vocal_embs.shape[2]} entries, each entry has {vocal_embs.shape[1]} dimensions"
        )
        if mode == "decode":
            vocal_toks = torch.zeros(batch_size, 1024, 1).long().to(device)
            bgm_toks = torch.zeros(batch_size, 1024, 1).long().to(device)
            input = torch.cat([vocal_toks,bgm_toks], dim=-1)
            with torch.no_grad():
                output = codec(input)
        else:
            with torch.no_grad():
                output = codec(sig)

        if mode == "reconstruct":
            save_dir = os.path.join(root_path, "codecs", "reconstruction_wav")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"mucodec_reconstruction.wav")
            torchaudio.save(
                save_path,
                output.cpu() if use_cuda else output,
                codec.orig_sample_rate,
            )
            print(f"{mode} mode has been saved to {save_path}")
        elif mode == "encode":
            print(f"{mode} mode, the output shape is {output[0].shape}")
        else:
            print(f"{mode} mode, the output shape is {output.shape}")

