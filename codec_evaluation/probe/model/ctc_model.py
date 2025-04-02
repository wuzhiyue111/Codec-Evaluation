import torch
import torch.nn as nn
from codec_evaluation.init_codecs import init_codec
from asr_decoder import CTCDecoder
from conformer import Conformer


class Ctc_Probe(nn.Module):
    def __init__(
        self,
        tokenizer,
        codec_dim,
        vocab_size=5000,
        dropout=0.1,
        conformer_head=8,
        **kwargs
    ):
        super(Ctc_Probe, self).__init__()
        self.tokenizer = tokenizer.tokenizer
        self.dropout = nn.Dropout(dropout)

        self.ctc_decoder = CTCDecoder()
        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.pad_token_id, reduction="mean", zero_infinity=True
        )
        self.automatic_optimization = False

        # if codec_dim >= 1024, use the projection layer will get bad performance
        if codec_dim < 1024:
            self.codec_projection = nn.Linear(codec_dim, 1024)
            input_dim = 1024
            self.use_projection = True
        else:
            self.codec_projection = nn.Identity()
            # self.codec_projection = nn.Linear(codec_dim, 1024)
            # input_dim = 1024
            input_dim = codec_dim
            self.use_projection = False
            # self.use_projection = True

        # assert input_dim % conformer_head == 0, "The dimension of the codec model must be divisible by the number of conformer heads"
        self.conformer = Conformer(
            dim=input_dim, # 1024
            depth=3,  # 3 blocks
            dim_head=input_dim // conformer_head,
            heads=conformer_head,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.0,
            ff_dropout=0.0,
            conv_dropout=0.0,
        )
        self.conformer_head = nn.Linear(input_dim, vocab_size)
    
    def forward(self, feature, feature_length, text):
        # feature: [B, D, T]
        feature = feature.transpose(1, 2)
        feature = self.dropout(feature)
        if self.use_projection:
            feature = self.codec_projection(feature)
        feature = self.conformer(feature)
        feature = self.conformer_head(feature)
        feature_logits_prob = torch.nn.functional.log_softmax(
            feature, dim=-1, dtype=torch.float32
        )  # [B, T, V]
        output = self.tokenizer(text, padding=True, return_tensors="pt")
        labels = output["input_ids"].to(feature.device)
        labels_mask = output["attention_mask"]
        labels_lengths = tuple(mask.sum().item() for mask in labels_mask)

        loss = self.criterion(feature_logits_prob.transpose(0, 1), labels, tuple(int(length.item()) for length in feature_length), labels_lengths)
        return loss

    def inference(self, feature):
        feature = feature.transpose(1, 2)
        if self.use_projection:
            feature = self.codec_projection(feature)
        feature = self.conformer(feature)
        feature = self.conformer_head(feature)
        feature_logits_prob = torch.nn.functional.log_softmax(
            feature, dim=-1, dtype=torch.float32
        )  # [B, T, V]
        return feature_logits_prob