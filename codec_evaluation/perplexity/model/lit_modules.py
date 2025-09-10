import pytorch_lightning as pl
from codec_evaluation.codecs.init_codecs import init_codec
from codec_evaluation.perplexity.model.decoder_only_100M_qwen import PPL_100M_ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from typing import Any
import torch

class PPL_lit_modules(pl.LightningModule):
    def __init__(self, 
                 ppl_model_config: Qwen2Config,
                 codec_name: str,
                 codec_ckpt_dir: str,
                 mode = "encode",
                 sample_rate: int = 24000, # audio
                 lm_head_nums: int = 1,
                 lr_scheduler_builder: Any = None,
                 optimizer_builder: Any = None,
                 accumulate_grad_batches: int = 1,
                 ):
        super(PPL_lit_modules, self).__init__()
        self.codec = init_codec(modelname = codec_name,
                        mode = mode,
                        sample_rate = sample_rate,
                        model_ckpt_dir = codec_ckpt_dir,
                        device = "cpu",
                        freeze = True)
        ppl_model_config.vocab_size = self.codec.vocab_size + 3
        ppl_model_config.pad_token_id = self.codec.vocab_size + 2
        self.model = PPL_100M_ForCausalLM(config = ppl_model_config,
                                          lm_head_nums = lm_head_nums,
                                          num_items_in_batch = accumulate_grad_batches)
        self.accumulate_grad_batches = accumulate_grad_batches

        self.ppl_model_config = ppl_model_config
        self.sos_id = self.codec.vocab_size
        self.eos_id = self.codec.vocab_size + 1
        self.lr_scheduler_builder = lr_scheduler_builder
        self.optimizer_builder = optimizer_builder
        self.codebook_size = self.codec.vocab_size

    def configure_optimizers(self):
        optimizer = self.optimizer_builder(self.model.parameters())
        lr_scheduler = self.lr_scheduler_builder(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler, 
                    "interval": "step"}
                }

    def data_process(self, input_ids, audio_lengths):
        """
            Process data using tensor parallelization operations.
        """
        B, T, K = input_ids.shape

        # 1. Calculate the actual sequence length for each sample
        input_ids_len = audio_lengths // self.codec.hop_length

        # 2. Determine the new lengths and max length after adding SOS and EOS
        new_lengths = input_ids_len + 2  # Added SOS and EOS
        max_len = new_lengths.max()

        # 3. Create target tensors and initialize with padding values
        # input_ids initialized to pad_token_id
        new_input_ids = torch.full((B, max_len, K), self.ppl_model_config.pad_token_id, dtype=torch.long, device=self.device)
        # labels initialized to -100
        labels = torch.full((B, max_len, K), -100, dtype=torch.long, device=self.device)

        # 4. Place SOS at the starting position [0] of all sequences
        new_input_ids[:, 0, :] = self.sos_id
        labels[:, 0, :] = self.sos_id

        # 5. Use scatter operation to efficiently place EOS at the end of each sequence
        # new_lengths - 1 is the position index for the EOS token
        eos_indices = (new_lengths - 1).view(B, 1, 1).expand(-1, -1, K)
        new_input_ids.scatter_(1, eos_indices, self.eos_id)
        labels.scatter_(1, eos_indices, self.eos_id)

        # 6. Create a mask to place the real tokens
        # arange(max_len) -> [0, 1, 2, ..., max_len-1]
        # .unsqueeze(0) -> [[0, 1, 2, ...]]
        # .expand(B, -1) -> [[0, 1, ...], [0, 1, ...], ...]
        # The shape of the mask is (B, max_len)
        mask_range = torch.arange(max_len, device=self.device).expand(B, -1)
        # input_ids_len.unsqueeze(1) -> [[len1], [len2], ...]
        # The mask marks positions from index 1 to real_len
        real_token_mask = (mask_range > 0) & (mask_range <= input_ids_len.unsqueeze(1))
        
        # 7. Place the original input_ids into the correct positions based on the mask
        # real_token_mask.unsqueeze(-1).expand(-1, -1, K) expands the mask to match the K dimension
        # real_token_mask[:, 1:T+1] ensures operations are only within the possible target area
        target_mask_for_input = real_token_mask[:, 1:T+1]
        
        new_input_ids[:, 1:T+1, :][target_mask_for_input] = input_ids[target_mask_for_input.squeeze(-1)]
        labels[:, 1:T+1, :][target_mask_for_input] = input_ids[target_mask_for_input.squeeze(-1)]

        return new_input_ids, labels, B

    def _step(self, batch, batch_idx):
        audios = batch["audio"]
        audio_lengths = batch["audio_length"]

        with torch.no_grad():
            input_ids, _ = self.codec(audios) # [B, T, K]
            input_ids = input_ids.to(torch.long)
            input_ids, labels, batch_size = self.data_process(input_ids, audio_lengths)
        logits_list, loss_list = self.model(input_ids = input_ids.clone(), labels = labels.clone()) # clone to avoid gradient vanishing

        # calculate valid token number (labels != -100)
        valid_tokens = (labels != -100).sum()
        valid_tokens = valid_tokens.clamp_min(1).to(loss_list[0].dtype).to(loss_list[0].device)

        return logits_list, loss_list, labels, batch_size, valid_tokens
    
    def log_loss_list(self, loss_list, stage, batch_size, valid_tokens):
        loss = torch.stack(loss_list).mean()
        normalized_loss_list = [codebook_loss.detach() * self.accumulate_grad_batches / valid_tokens for codebook_loss in loss_list]

        if stage == "train":
            self.log(f"{stage}/loss_mean", loss.detach() / valid_tokens, on_step = True, on_epoch = True, prog_bar = True, logger = True, sync_dist=True, batch_size = batch_size)
        else:
            self.log(f"{stage}_loss_mean", loss.detach() / valid_tokens, on_step = False, on_epoch = True, prog_bar = True, logger = True, sync_dist=True, batch_size = batch_size)

        for i, tmp_loss in enumerate(normalized_loss_list):
            tmp_log_loss = tmp_loss.detach()
            if stage == "train":
                self.log(f"{stage}/loss_codebook_{i}", tmp_log_loss, on_step = True, on_epoch = True, prog_bar = False, logger = True, sync_dist=True, batch_size = batch_size)
            else:
                self.log(f"{stage}_loss_codebook_{i}", tmp_log_loss, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist=True, batch_size = batch_size)
        return loss

    def log_ppl_list(self, loss_list, stage, batch_size):
        ppl_list = [torch.exp(loss) / (self.codebook_size / 1024) for loss in loss_list]
        loss = torch.stack(loss_list).mean()
        ppl = torch.exp(loss) / (self.codebook_size / 1024)
        if stage != "train":
            self.log(f"{stage}_ppl_mean", ppl, on_step = False, on_epoch = True, prog_bar = True, logger = True, sync_dist=True, batch_size = batch_size)
        
        for i, tmp_ppl in enumerate(ppl_list):
            if stage != "train":
                self.log(f"{stage}_ppl_codebook_{i}", tmp_ppl, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist=True, batch_size = batch_size)
        
        return ppl

    def on_before_optimizer_step(self, optimizer):
        """
        在优化器更新前计算并记录梯度范数。
        Compute and log gradient norms before the optimizer updates weights.
        """
        # 确保在训练阶段才记录
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            # 我们只关心需要更新梯度的参数
            params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True and param.grad == None:
                    print(f"name: {name} param grad == None")
                elif param.requires_grad == True and param.grad != None:
                    params.append(param)
            # params = [p for p in self.model.parameters() if p.grad is not None]
            if not params:
                return

            # 计算总的 L2 范数 (与梯度裁剪的方式相同)
            # Calculate the total L2 norm (same way as gradient clipping)
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)

            # 记录总范数
            # Log the total norm
            self.log('train/grad_norm_total', total_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def calculate_and_log_acc(self, logits_list, labels, topk: list[int] = [1, 5], stage: str = "train", batch_size: int = 1):
        accuracy_list = []
        for i in range(self.model.lm_head_nums):
            accuracy_list.append(self.get_accuracy(logits_list[i], labels[:, :, i], topk = topk))
        
        for i in range(len(topk)):
            acc = torch.mean(torch.stack([accuracy_list[j][i] for j in range(self.model.lm_head_nums)]))
            self.log(f"{stage}/id_acc_{topk[i]}", acc, on_step = True, on_epoch = True, prog_bar = True, logger = True, sync_dist=True, batch_size = batch_size)

    def training_step(self, batch, batch_idx):
        logits_list, loss_list, labels, batch_size, valid_tokens = self._step(batch, batch_idx)
        self.calculate_and_log_acc(logits_list, labels, topk = [1, 5, 10, 50, 100, 200], stage = "train", batch_size = batch_size)

        loss = self.log_loss_list(loss_list, "train", batch_size, valid_tokens)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits_list, loss_list, labels, batch_size, token_length = self._step(batch, batch_idx)
        
        loss = torch.stack(loss_list).mean()
        ppl = torch.exp(loss)
        
        topk = [1, 5, 10, 50, 100, 200]
        accuracy_list = []
        for i in range(self.model.lm_head_nums):
            accuracy_list.append(self.get_accuracy(logits_list[i], labels[:, :, i], topk=topk))
            
        outputs = {
            "val_loss": loss,
            "val_ppl": ppl,
            "val_accuracy_list": accuracy_list
        }
        return outputs

    def test_step(self, batch, batch_idx):
        logits_list, loss_list, labels, batch_size, token_length = self._step(batch, batch_idx)
        
        loss = torch.stack(loss_list).mean()
        ppl = torch.exp(loss)
        
        topk = [1, 5, 10, 50, 100, 200]
        accuracy_list = []
        for i in range(self.model.lm_head_nums):
            accuracy_list.append(self.get_accuracy(logits_list[i], labels[:, :, i], topk=topk))
            
        outputs = {
            "test_loss": loss,
            "test_ppl": ppl,
            "test_accuracy_list": accuracy_list
        }
        return outputs

    def get_accuracy(
        self,
        logits,
        labels,
        ignore_index = [-100],
        topk: list[int] = [1, 5],
    ):
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        accuracy_list = []

        # 创建掩码，标记不需要忽略的位置
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
        for ignore_id in ignore_index:
            valid_mask &= labels != ignore_id

        for k in topk:
            _, indices = logits.topk(k, dim=-1)
            correct = indices.eq(labels.unsqueeze(-1))
            for ignore_id in ignore_index:
                correct[labels == ignore_id] = 0
            correct = correct.sum()
            # 使用valid_mask计算有效标签的数量
            accuracy = correct / valid_mask.sum()
            accuracy_list.append(accuracy)
        return accuracy_list

        
        
