import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Dict
from codec_evaluation.codecs.init_codecs import init_codec
from conformer import Conformer 
from codec_evaluation.probe.model.ctc_model import Ctc_Probe
from torchmetrics.text import WordErrorRate, CharErrorRate
from asr_decoder import CTCDecoder


class CodecCTCProbe(pl.LightningModule):
    def __init__(
        self,
        codec_name: str,
        mode: str,
        sample_rate: int,
        model_ckpt_dir: str,
        tokenizer: Any = None,
        probe_model_builder: Any = None,
        optimizer_builder: Any= None,
        lr_scheduler_builder: Any = None,
        feature_extractor_config_path: Any = None,
        teacher_ckpt_path: Any = None,
        
    ):
        super().__init__()
        self.codec_name = codec_name
        self.mode = mode
        self.sample_rate = sample_rate
        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.tokenizer = tokenizer
        self.codec_model = init_codec(
            modelname=codec_name,
            mode=mode,
            sample_rate=sample_rate,
            model_ckpt_dir=model_ckpt_dir,
            device="cpu",
            freeze=True,
            feature_extractor_config_path=feature_extractor_config_path,
            teacher_ckpt_path = teacher_ckpt_path
        )
        
        self.codec_dim = self.codec_model.dim * 2 if self.codec_name == 'semanticodec' else self.codec_model.dim

        self.probe_model: Ctc_Probe = probe_model_builder(self.tokenizer)

        self.ctc_decoder = CTCDecoder()

        self.test_step_outputs = []

        self.val_wer = WordErrorRate()
        self.val_cer = CharErrorRate()
        self.test_wer = WordErrorRate()
        self.test_cer = CharErrorRate()

    def extract_feature(self, waveforms, expect_lenth: torch.Tensor = None):
        """
            extract features from codec
            waveforms: [B, T]
            return: [B*n_segments, D, T]
        """
        length = torch.ones(waveforms.shape[0])

        input_ids, _ = self.codec_model(waveforms) # [B, T, K]
        input_ids = input_ids.to(torch.long).clone()
        
        if self.codec_name == 'semanticodec':
            assert expect_lenth is not None, "expect_lenth is required for semanticodec"
            max_length = max(expect_lenth).item()
            input_ids = input_ids[:, :int(max_length), :]

        return input_ids

    def _get_feature_lengths(self, audio_lengths: torch.Tensor):
        
        if self.codec_model.orig_sample_rate != self.sample_rate:
            ratio = self.codec_model.orig_sample_rate / self.sample_rate
            feature_lengths = (audio_lengths * ratio) // self.codec_model.hop_length
        else:
            feature_lengths = audio_lengths // self.codec_model.hop_length
        
        if self.codec_name == "hubert":
            feature_lengths -= 1
        
        return feature_lengths.to(torch.int32)

    def _shared_step(self, batch):
        
        waveforms = batch["audio"]
        texts = batch["text"]
        audio_lengths = batch["audio_length"]

        feature_lengths = self._get_feature_lengths(audio_lengths)
        input_ids = self.extract_feature(waveforms, feature_lengths)
        
        loss, logits = self.probe_model(input_ids, feature_lengths, texts)

        return loss, logits, texts, feature_lengths
    
    def _ctc_decode_batch(self, logits: torch.Tensor, feature_lengths: torch.Tensor):

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_texts = []

        for i in range(log_probs.shape[0]):
            current_log_probs = log_probs[i, :feature_lengths[i], :]
            decoded_result = self.ctc_decoder.ctc_greedy_search(current_log_probs, is_last=True)
            self.ctc_decoder.reset() 

            text = self.tokenizer.decode(decoded_result["tokens"], skip_special_tokens=True)
            pred_texts.append(text)
            
        return pred_texts


    def training_step(self, batch, batch_idx):
        loss, _, _ , _= self._shared_step(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            
            print(f"Skipping bad loss on batch {batch_idx}: {loss.item()}")
            return None 

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        loss, logits, ground_truth_texts, feature_lengths = self._shared_step(batch)

        # _, pred_texts = self.ctc_greedy_decode(logits, feature_lengths)
        pred_texts = self._ctc_decode_batch(logits, feature_lengths)

        # 更新torchmetrics指标
        self.val_wer.update(pred_texts, ground_truth_texts)
        self.val_cer.update(pred_texts, ground_truth_texts)

        # 在epoch结束时会自动计算和记录
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_wer", self.val_wer, on_epoch=True, prog_bar=True)
        self.log("val_cer", self.val_cer, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        
        _, logits, ground_truth_texts, feature_lengths = self._shared_step(batch)
        pred_texts = self._ctc_decode_batch(logits, feature_lengths)
        
        self.test_wer.update(pred_texts, ground_truth_texts)
        self.test_cer.update(pred_texts, ground_truth_texts)

        # 在test_epoch_end时会自动计算和记录最终结果
        self.log("test_wer", self.test_wer, on_epoch=True)
        self.log("test_cer", self.test_cer, on_epoch=True)
        self.test_step_outputs = {"wer": self.test_wer, "cer": self.test_cer}

    def on_test_epoch_end(self):
       
        final_wer = self.test_wer.compute()
        final_cer = self.test_cer.compute()
        self.log("test_wer", final_wer)
        self.log("test_cer", final_cer)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""

        optimizer = self.optimizer_builder(self.parameters())
        if self.lr_scheduler_builder:
            scheduler = self.lr_scheduler_builder(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", 
                },
            }
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if name.startswith("codec_model."):
                state_dict.pop(name)
