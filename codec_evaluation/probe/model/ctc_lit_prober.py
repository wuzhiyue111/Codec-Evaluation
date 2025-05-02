from pytorch_lightning import LightningModule
from codec_evaluation.codecs.init_codecs import init_codec
from codec_evaluation.utils.utils import cut_or_pad
import torch
from codec_evaluation.utils.logger import RankedLogger
import os
from typing import Dict, Any
from asr_decoder import CTCDecoder
from jiwer import wer, cer
from codec_evaluation.probe.model.ctc_model import Ctc_Probe
from codec_evaluation.reconstruction_eval.utils import transform_text_list_for_wer, transform_text_list_for_cer
from einops import rearrange

logger = RankedLogger(__name__, rank_zero_only=True)

class CtcLitProber(LightningModule):
    def __init__(
        self,
        codec_name: str,
        sample_rate: int,
        model_ckpt_dir: str,
        language: str,
        mode: str = "quantized_emb",
        probe_model_builder: Any = None,
        optimizer_builder: Any = None,
        lr_scheduler_builder: Any = None,
        accumulate_grad_batches: int = 1,
    ):
        super(CtcLitProber, self).__init__()
        self.codec_model = init_codec(
            modelname=codec_name,
            mode=mode,
            sample_rate=sample_rate,
            model_ckpt_dir=model_ckpt_dir,
            device="cpu",
            freeze=True
        )
        if codec_name == 'semanticodec':
            self.dim = self.codec_model.dim * 2
        else:
            self.dim = self.codec_model.dim

        logger.info(f"{codec_name} dim: {self.dim}")
        self.probe_model: Ctc_Probe = probe_model_builder(
            codec_dim = self.dim)
        self.language = language
        self.codec_name = codec_name
        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.ctc_decoder = CTCDecoder()
        self.test_step_outputs = []
        self.sample_rate = sample_rate

    def extract_feature(self, waveforms, expect_lenth: torch.Tensor = None):
        """
            extract features from codec
            waveforms: [B, T]
            return: [B*n_segments, D, T]
        """
        length = torch.ones(waveforms.shape[0])
        all_features = self.codec_model(waveforms, length)

        if self.codec_name == 'semanticodec':
            assert expect_lenth is not None, "expect_lenth is required for semanticodec"
            if all_features.dim() == 4:
                all_features = rearrange(all_features, 'b d c t -> b (d c) t')
            max_length = max(expect_lenth).item()
            all_features = all_features[:, :, :int(max_length)]

        return all_features
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Do not save codec
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if "codec_model" in name:
                state_dict.pop(name)

    def step(self, batch):
        audio = batch["audio"]
        text = batch["text"]
        audio_length = batch["audio_length"]
        batch_size = audio.shape[0]

        # consider the resample function of codec, libriTTS dataset is 24000 sample rate
        if self.codec_model.orig_sample_rate != self.sample_rate:
            feature_length = audio_length * (self.codec_model.orig_sample_rate / self.sample_rate) // self.codec_model.hop_length
        else:
            feature_length = audio_length // self.codec_model.hop_length
        audio_features = self.extract_feature(audio, feature_length)
        loss = self.probe_model(audio_features, feature_length, text)
        return loss, batch_size

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        loss, batch_size = self.step(batch)

        # 检查loss是否正常
        if torch.isnan(loss) or torch.isinf(loss):
            logger.info(f"Skipping bad loss: {loss.item()}")
            return {"loss": torch.tensor(0.0).to(self.device).requires_grad_(True).detach()}
        
        # 梯度累积
        self.manual_backward(loss / self.accumulate_grad_batches)
        if (batch_idx+1) % self.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
    
    def on_train_batch_end(self, outputs: torch.Tensor | os.Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if (batch_idx+1) % (self.accumulate_grad_batches * 10) == 0: # per 10 steps empty cache
            torch.cuda.empty_cache()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def configure_optimizers(self):
        """Configure the optimizer and scheduler."""
        optimizer = self.optimizer_builder(self.probe_model.parameters())
        if self.lr_scheduler_builder is not None:
            lr_scheduler = self.lr_scheduler_builder(optimizer)
            return {"optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "step",
                    }}
        else:
            return optimizer
        
    def post_process_text_for_wer(self, text_list):
        # 如果输入是list，先合并成字符串
        # 过滤掉特殊token
        filtered = [word for word in text_list if word not in ["", "<pad>", "</s>", "<unk>", "[PAD]", "[UNK]"]]
        text = " ".join(filtered)
        text = transform_text_list_for_wer([text])[0]
        return text
    
    def post_process_text_for_cer(self, text_list):
        # 如果输入是list，先合并成字符串
        # 过滤掉特殊token
        filtered = [word for word in text_list if word not in ["", "<pad>", "</s>", "<unk>", "[PAD]", "[UNK]"]]
        text = " ".join(filtered)
        text = transform_text_list_for_cer([text])[0]
        return text

    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        text = batch["text"]

        # semantic will pad the audio to multiple of 10.24
        if self.codec_name == 'semanticodec':
            audio_length = batch["audio_length"]
            if self.codec_model.orig_sample_rate != self.sample_rate:
                feature_length = audio_length * (self.codec_model.orig_sample_rate / self.sample_rate) // self.codec_model.hop_length
            else:
                feature_length = audio_length // self.codec_model.hop_length
            audio_features = self.extract_feature(audio, feature_length)
        else:
            audio_features = self.extract_feature(audio)

        feature_logits_prob = self.probe_model.inference(audio_features)

        wer_list = []
        result_list = []
        cer_list = []
        for i in range(len(text)):
            result = self.ctc_decoder.ctc_greedy_search(feature_logits_prob[i], is_last=True)
            self.ctc_decoder.reset()
            pred_text = [self.probe_model.tokenizer.decode(r) for r in result["tokens"]]
            pred_text_for_wer = self.post_process_text_for_wer(pred_text)
            labels_text_for_wer = self.post_process_text_for_wer([text[i]])
            pred_text_for_cer = self.post_process_text_for_cer(pred_text)
            labels_text_for_cer = self.post_process_text_for_cer([text[i]])
            wer_list.append(wer(labels_text_for_wer, pred_text_for_wer))
            result_list.append({"pred_text": pred_text_for_wer, "labels_text": labels_text_for_wer})
            cer_list.append(cer(labels_text_for_cer, pred_text_for_cer))
        wer_result = sum(wer_list) / len(wer_list)
        cer_result = sum(cer_list) / len(cer_list)

        self.test_step_outputs.append({"wer": wer_result, "cer": cer_result, "result": result_list})
    
    def on_test_epoch_end(self):
        wer_list = []
        result = []
        cer_list = []
        for output in self.test_step_outputs:
            wer_list.append(output["wer"])
            for r in output["result"]:
                result.append(r)
            cer_list.append(output["cer"])
        avg_wer = sum(wer_list) / len(wer_list)
        avg_cer = sum(cer_list) / len(cer_list)

        if self.language == 'en':
            self.test_step_outputs = {"wer": avg_wer, "cer": avg_cer}
        else:
            self.test_step_outputs = {"cer": avg_cer}