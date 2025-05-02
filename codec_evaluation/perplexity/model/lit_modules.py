import pytorch_lightning as pl
import torch.nn as nn
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
                 sample_rate: int = 24000, # libritts
                 lm_head_nums: int = 1,
                 lr_scheduler_builder: Any = None,
                 optimizer_builder: Any = None,
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
                                          lm_head_nums = lm_head_nums)


        self.ppl_model_config = ppl_model_config
        self.sos_id = self.codec.vocab_size
        self.eos_id = self.codec.vocab_size + 1
        self.lr_scheduler_builder = lr_scheduler_builder
        self.optimizer_builder = optimizer_builder

    def configure_optimizers(self):
        optimizer = self.optimizer_builder(self.model.parameters())
        lr_scheduler = self.lr_scheduler_builder(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler, 
                    "interval": "step"}
                }

    def data_process(self, input_ids, audio_lengths):
        input_ids_len = audio_lengths // self.codec.hop_length

        B, T, K = input_ids.shape
        input_ids_list = []
        labels_list = []
        for i in range(B):
            real_labels_len = input_ids_len[i]
            real_labels = input_ids[i, :real_labels_len, :]
            ignore_len = T - real_labels_len
            if ignore_len > 0:
                ignore_labels = torch.full((ignore_len, K), -100, dtype=torch.long, device=self.device)
                
            special_sos_labels = torch.full((1, K), self.sos_id, dtype=torch.long, device=self.device)
            special_eos_labels = torch.full((1, K), self.eos_id, dtype=torch.long, device=self.device)
            if ignore_len > 0:
                labels = torch.cat([special_sos_labels, real_labels, special_eos_labels, ignore_labels], dim=0)
            else:
                labels = torch.cat([special_sos_labels, real_labels, special_eos_labels], dim=0)
            labels_list.append(labels)

            if ignore_len > 0:
                pad_ids = torch.full((ignore_len, K), self.ppl_model_config.pad_token_id, dtype=torch.long, device=self.device)
                input_ids_list.append(torch.cat([special_sos_labels, real_labels, special_eos_labels, pad_ids], dim=0))
            else:
                input_ids_list.append(torch.cat([special_sos_labels, real_labels, special_eos_labels], dim=0))

        return torch.stack(input_ids_list, dim=0), torch.stack(labels_list, dim=0), B

    def _step(self, batch, batch_idx):
        audios = batch["audio"]
        audio_lengths = batch["audio_length"]

        with torch.no_grad():
            input_ids, _ = self.codec(audios) # [B, T, K]
            input_ids = input_ids.to(torch.long)
            input_ids, labels, batch_size = self.data_process(input_ids, audio_lengths)
        logits_list, loss_list = self.model(input_ids = input_ids, labels = labels)
        return logits_list, loss_list, labels, batch_size
    
    def log_loss_list(self, loss_list, stage, batch_size):
        loss = torch.stack(loss_list).mean()
        if stage == "train":
            self.log(f"{stage}/loss_mean", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, batch_size = batch_size)
        else:
            self.log(f"{stage}_loss_mean", loss, on_step = False, on_epoch = True, prog_bar = True, logger = True, sync_dist=True, batch_size = batch_size)

        for i, tmp_loss in enumerate(loss_list):
            if stage == "train":
                self.log(f"{stage}/loss_codebook_{i}", tmp_loss, on_step = True, on_epoch = True, prog_bar = False, logger = True, batch_size = batch_size)
            else:
                self.log(f"{stage}_loss_codebook_{i}", tmp_loss, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist=True, batch_size = batch_size)
        return loss

    def calculate_and_log_acc(self, logits_list, labels, topk: list[int] = [1, 5], stage: str = "train", batch_size: int = 1):
        accuracy_list = []
        for i in range(self.model.lm_head_nums):
            accuracy_list.append(self.get_accuracy(logits_list[i], labels[:, :, i], topk = topk))
        
        for i in range(len(topk)):
            acc = torch.mean(torch.stack([accuracy_list[j][i] for j in range(self.model.lm_head_nums)]))
            self.log(f"{stage}/id_acc_{topk[i]}", acc, on_step = True, on_epoch = True, prog_bar = True, logger = True, batch_size = batch_size)

    def training_step(self, batch, batch_idx):
        logits_list, loss_list, labels, batch_size = self._step(batch, batch_idx)
        self.calculate_and_log_acc(logits_list, labels, topk = [1, 5, 10, 50, 100, 200], stage = "train", batch_size = batch_size)

        loss = self.log_loss_list(loss_list, "train", batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits_list, loss_list, labels, batch_size = self._step(batch, batch_idx)
        self.calculate_and_log_acc(logits_list, labels, topk = [1, 5, 10, 50, 100, 200], stage = "val", batch_size = batch_size)

        loss = self.log_loss_list(loss_list, "val", batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        logits_list, loss_list, labels, batch_size = self._step(batch, batch_idx)
        self.calculate_and_log_acc(logits_list, labels, topk = [1, 5, 10, 50, 100, 200], stage = "test", batch_size = batch_size)

        loss = self.log_loss_list(loss_list, "test", batch_size)
        return loss

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

        
        
