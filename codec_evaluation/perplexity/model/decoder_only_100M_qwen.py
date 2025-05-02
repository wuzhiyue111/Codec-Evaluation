from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
import torch.nn as nn
from transformers.loss.loss_utils import ForCausalLMLoss
from functools import partial
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

class PPL_100M_Qwen_Model(Qwen2Model):
    def __init__(self, config: Qwen2Config, lm_head_nums: int):
        super().__init__(config)
        self.lm_head_nums = lm_head_nums

        embed_tokens_list = []
        for _ in range(lm_head_nums):
            embed_tokens_list.append(nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id))
        self.embed_tokens = nn.ModuleList(embed_tokens_list)

    def forward(self, 
                input_ids = None, 
                attention_mask=None, 
                position_ids=None, 
                inputs_embeds=None,
                past_key_values=None, 
                use_cache=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None,  
                **kwargs):
        return super().forward(input_ids = input_ids, 
                               attention_mask = attention_mask, 
                               position_ids = position_ids, 
                               inputs_embeds = inputs_embeds,
                               past_key_values = past_key_values, 
                               use_cache = use_cache, 
                               output_attentions = output_attentions, 
                               output_hidden_states = output_hidden_states, 
                               return_dict = return_dict, **kwargs)

class PPL_100M_ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config, lm_head_nums: int):
        super().__init__()
        self.model = PPL_100M_Qwen_Model(config, lm_head_nums)
        self.lm_head_nums = lm_head_nums
        self.lm_head_list = []
        for _ in range(lm_head_nums):
            self.lm_head_list.append(nn.Linear(config.hidden_size, config.vocab_size))
        self.lm_head = nn.ModuleList(self.lm_head_list)
        self.lm_input_proj = nn.Linear(config.hidden_size * lm_head_nums, config.hidden_size, bias = False)
        self.loss_fn = partial(ForCausalLMLoss, vocab_size = config.vocab_size)

    def forward(self, 
                input_ids,
                labels,
                attention_mask = None, 
                position_ids = None, 
                inputs_embeds = None, 
                past_key_values = None, 
                use_cache = None, 
                output_attentions = None, 
                output_hidden_states = None, 
                return_dict = None, 
                **kwargs):
        # input_ids: (batch_size, seq_len, K), K is the number of lm_heads
        
        if input_ids is None:
            raise ValueError("input_ids should't be None")
        
        if labels is None:
            raise ValueError("labels should't be None")
        
        input_embeds_list = []
        for i in range(self.lm_head_nums):
            input_embeds_list.append(self.model.embed_tokens[i](input_ids[:, :, i]))
        inputs_embeds = torch.cat(input_embeds_list, dim = 2)
        inputs_embeds = self.lm_input_proj(inputs_embeds)

        outputs = self.model(input_ids = None, 
                             attention_mask = attention_mask, 
                             position_ids = position_ids, 
                             inputs_embeds = inputs_embeds, 
                             past_key_values = past_key_values, 
                             use_cache = use_cache, 
                             output_attentions = output_attentions, 
                             output_hidden_states = output_hidden_states, 
                             return_dict = return_dict, 
                             **kwargs)
        
        hidden_states = outputs.last_hidden_state

        loss_list = []
        logits_list = []
        for i in range(self.lm_head_nums):
            logits = self.lm_head[i](hidden_states)
            loss = self.loss_fn(logits, labels[:, :, i])
            loss_list.append(loss)
            logits_list.append(logits)

        return logits_list, loss_list
