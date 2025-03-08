from pytorch_lightning import LightningModule
from typing import Any
from codec_evaluation.init_codecs import init_codec
from codec_evaluation.utils.utils import cut_or_pad
import torch
from codec_evaluation.utils.logger import RankedLogger
import os
logger = RankedLogger(__name__, rank_zero_only=True)

class CtcLitProber(LightningModule):
    def __init__(
        self,
        codec_name: str,
        sample_rate: int,
        model_ckpt_dir: str,
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
        self.dim = self.codec_model.dim
        self.probe_model = probe_model_builder(
            codec_dim = self.dim)
        self.codec_name = codec_name
        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches

    def forward(self, waveforms):
        pass

    def extract_feature(self, waveforms):
        """
            extract features from codec
            waveforms: [B, T]
            return: [B*n_segments, D, T]
        """
        length = torch.ones(waveforms.shape[0])
        all_features = self.codec_model(waveforms, length)

        if self.codec_name == 'semanticodec':
            if all_features.dim() == 4:
                split_feature = all_features.unbind(dim=2) # [B*n_segments, D, 2, T]
            else:
                split_feature = all_features.chunk(2, dim=1) # [tensor[B*n_segments, D, T]、tensor[B*n_segments, D, T]]

            all_features = torch.cat(split_feature, dim=0)  # [2*B*n_segments, D, T]

        return all_features

    def step(self, batch):
        audio = batch["audio"]
        text = batch["text"]
        audio_length = batch["audio_length"]
        batch_size = audio.shape[0]
        feature_length = audio_length // self.codec_model.model.hop_length
        audio_features = self.extract_feature(audio)
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