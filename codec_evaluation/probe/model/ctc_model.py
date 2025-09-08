import torch
import torch.nn as nn
from conformer import Conformer


class Ctc_Probe(nn.Module):
    def __init__(
        self,
        tokenizer: str, 
        vocab_size: int,
        codec_vocab_size: int,
        lm_head_nums: int,
        conformer_depth: int = 3,
        conformer_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super(Ctc_Probe, self).__init__()

        self.vocab_size = vocab_size
        self.conformer_depth = conformer_depth
        self.conformer_heads = conformer_heads
        self.dropout = dropout
        self.codec_vocab_size = codec_vocab_size
        self.tokenizer = tokenizer
        self.lm_head_nums = lm_head_nums

        self.embed_tokens = nn.ModuleList([nn.Embedding(num_embeddings=codec_vocab_size + 1,
                                                        embedding_dim=1024,
                                                        padding_idx=codec_vocab_size
                                                    )
                                            for _ in range(lm_head_nums)
                                        ])
        
        self.dropout = nn.Dropout(self.dropout)
        
        self.feature_projection = nn.Linear(1024*lm_head_nums, 1024)

        self.conformer = Conformer(
            dim=1024,
            depth=self.conformer_depth,
            dim_head=1024 // self.conformer_heads,
            heads=self.conformer_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.0,
            ff_dropout=0.0,
            conv_dropout=0.0,
        )
        self.ctc_head = nn.Linear(1024, self.vocab_size)

        self.criterion = nn.CTCLoss(
            blank=self.tokenizer.tokenizer.pad_token_id, reduction="mean", zero_infinity=True
        )
    
    def forward(self, input_ids, feature_lengths, texts):
        # feature: [B, D, T]
        input_embeds_list = []
        for i in range(self.lm_head_nums):
            input_embeds_list.append(self.embed_tokens[i](input_ids[:, :, i]))
        features = torch.cat(input_embeds_list, dim = 2)
        features = self.feature_projection(features)
        
        features = self.dropout(features)
        conformer_output = self.conformer(features)
        logits = self.ctc_head(conformer_output) # [B, T, V]

        tokenized_output = self.tokenizer(text=texts, padding="longest", return_tensors="pt", add_special_tokens=False)
        labels = tokenized_output.input_ids
        label_lengths = tokenized_output.attention_mask.sum(dim=-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1) # [T, B, V]
        loss = self.criterion(log_probs, labels, feature_lengths, label_lengths)

        return loss, logits
