from typing import Sequence, Optional, Union
import math
import torch
import codec_evaluation
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from codec_evaluation.codecs.YuE.quantization  import ResidualVectorQuantizer
from transformers import AutoModel
from codec_evaluation.codecs.YuE.RepCodec.repcodec.modules.encoder import Encoder
from codec_evaluation.codecs.YuE.RepCodec.repcodec.modules.decoder import Decoder
from codec_evaluation.codecs.YuE.descriptaudiocodec.dac.model import dac as dac2
root_path = codec_evaluation.__path__[0]

class SoundStream(nn.Module):
    """ SoundStream model or EnCodec model.
    
    Args:
        n_filters (int): n_filters (int): Base width for the model.
        D (int): Intermediate representation dimension.
        target_bandwidths (Sequence[int]): Target bandwidths in K-bits/second.
        ratios (Sequence[int]): downsampling factors, whose multiplication is the hop size.
        sample_rate (int): wave sampling rate.
        bins (int): number of code words in a codebook.
        normalize (bool): audio normalization.

    """
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        # target_bandwidths: Sequence[Union[int, float]] = [0.5, 1, 1.5, 2, 4, 6],
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2], #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        normalize: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        # total nb of codebooks, e.g., 6Kb/s, sr=16000 and hop_length=320 => nq = 12
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios)) # 50 Hz
        self.bits_per_codebook = int(math.log2(bins)) # 1024 => 10
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate
        self.encoder = dac2.Encoder(64,ratios,D)
        self.encoder_semantic = Encoder(input_channels=768,encode_channels=768)
        self.decoder_semantic = Decoder(code_dim=768,output_channels=768,decode_channels=768)
        # out_D=D+768
        self.quantizer = ResidualVectorQuantizer(dimension=D+768, n_q=n_q, bins=bins)
        self.decoder_2 = dac2.Decoder(D,1024,ratios,)

        self.is_semantic= True 
        if self.is_semantic:
            self.semantic_model = AutoModel.from_pretrained(os.path.join(root_path, "codecs", "YuE", "semantic_ckpts"))
            self.semantic_model.eval()

        self.fc_prior = nn.Linear(D+768, D+768 )
        self.fc_post1= nn.Linear( D+768, 768 )
        self.fc_post2= nn.Linear( D+768,  D)

    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def calculate_rec_loss(self, rec, target):  
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x ):
        x= x[:,0,:]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True) .hidden_states
        target = torch.stack(target, dim=1)#.transpose(-1, -2)#.flatten(start_dim=1, end_dim=2)
        target = target.mean(1)   

        return target

 
    def forward(self, x: torch.Tensor, bw: int):
        e_semantic_input = self.get_regress_target_whisper(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)
        e= torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        quantized_semantic = self.fc_post1(quantized.transpose(1, 2)).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)
        o = self.decoder_2(quantized_acoustic)
        o_semantic = self.decoder_semantic(quantized_semantic )
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(),o_semantic)

        return o, commit_loss, semantic_recon_loss,None


    def encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        bw = target_bw
        e_semantic_input = self.get_regress_target(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            e_acoustic = self.encoder(torch.transpose(F.pad(x[:,0,:], (160, 160)).unsqueeze(0), 0, 1))
        e= torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        return codes, e

    def get_embed(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.decode(codes)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)
        o = self.decoder_2(quantized_acoustic)
        return o

