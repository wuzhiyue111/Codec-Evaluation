import torch
import torchmetrics 
import pytorch_lightning as pl
from einops import rearrange
from codec_evaluation.codecs.init_codecs import init_codec
from typing import Any

class Prober(pl.LightningModule):
    """
        self.num_outputs: Number of label categories;
        self.codec: Initialied codec;
        self.token_rate: The number of tokens per sec;
        self.in_ch: The number of channels for deep convolution;
        self.dim: The D dimension of the codec output;
        task : [multiclass, regression, multilabel]
        metrics:
            multiclass : acc, f1
            regression : r2, arousal_r2, valence_r2
            multilabel : ap, aucroc
    """
    def __init__(self, 
                 codec_name: str,
                 sample_rate: int,
                 model_ckpt_dir: str,
                 task: str,
                 num_outputs: int,
                 mode: str = 'quantized_emb',
                 target_sec: int = 30,
                 probe_model_builder: Any = None,
                 optimizer_builder: Any = None,
                 lr_scheduler_builder: Any = None,
                 feature_extractor_config_path:Any = None,
                 teacher_ckpt_path: Any = None,
                 ):
        """
            codec_name must in ['dac', 'encodec', 'mimi', 'semanticodec', 'speechtokenizer', wavtokenizer]
            sample_rate: the audio sample_rate when you are training the probe model
            mode must in ['quantized_emb', 'unquantized_emb']
        """
        super(Prober, self).__init__()  
        self.codec = init_codec(modelname = codec_name,      
                                sample_rate = sample_rate, 
                                mode = mode, 
                                model_ckpt_dir = model_ckpt_dir,
                                device = 'cpu', 
                                freeze = True,
                                feature_extractor_config_path = feature_extractor_config_path,
                                teacher_ckpt_path = teacher_ckpt_path)
        self.codec_name = codec_name
        self.sample_rate = sample_rate
        
        if codec_name == "semanticodec":
            self.dim = self.codec.dim * 2
            self.token_rate = self.codec.token_rate / 2
        else:
            self.dim = self.codec.dim  
            self.token_rate = self.codec.token_rate

        self.audio_length = target_sec * self.sample_rate  
        self.feature_length = self.audio_length // self.codec.hop_length

        if self.codec.orig_sample_rate != self.sample_rate:
            self.feature_length = self.audio_length * (self.codec.orig_sample_rate / self.sample_rate) // self.codec.hop_length
        
        if self.codec_name == "hubert":
            self.feature_length = self.feature_length - 1

        self.probe_model = probe_model_builder(
            codec_dim = self.dim,
            target_T = int(self.feature_length))
        
        self.num_outputs = num_outputs
        self.task = task
        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.init_metrics()  
        self.test_step_outputs = []
    
    def extract_feature(self, waveforms, expect_lenth: torch.Tensor = None):
        """
            extract features from codec
            waveforms: [B, T]
            return: [B* split_count, D, T]
        """
        length = torch.ones(waveforms.shape[0])
        all_features = self.codec(waveforms, length)
        # import pdb; pdb.set_trace()
        if self.codec_name == 'semanticodec' or self.codec_name == 'mimi' or self.codec_name == 'qwen2audioencoder':
            assert expect_lenth is not None, "expect_lenth is required for semanticodec"
            if all_features.dim() == 4:
                all_features = rearrange(all_features, 'b d c t -> b (d c) t')
            max_length = expect_lenth
            all_features = all_features[:, :, :int(max_length)]

        return all_features

        # consider the resample function of codec, libriTTS dataset is 24000 sample rate

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, batch_size, _, _ = self.step(batch)

        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, batch_size, labels_pred, labels = self.step(batch)
        self.log('val_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_and_log_metrics("val", labels_pred, labels)

        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        
        loss, batch_size, labels_pred, labels = self.step(batch)

        self.update_and_log_metrics("test", labels_pred, labels)
    
    def step(self, batch):
        audio = batch["audio"]
        labels = batch["labels"]
        total_audio_length = batch["audio_length"]

        batch_size = labels.shape[0]

        audio_features = self.extract_feature(audio, int(self.feature_length))
        split_count = torch.round( max(total_audio_length) / self.audio_length).item()

        loss, labels_pred = self.probe_model(audio_features, labels, split_count)
        return loss, batch_size, labels_pred, labels


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
    
    def init_metrics(self):
        """
            metricd intialization
            check torchmetrics version == 1.4.1
        """
        if self.task == 'multilabel':
            self.val_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=self.num_outputs, ignore_index=-100)
            self.val_aucroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_outputs, ignore_index=-100)
            self.test_ap = torchmetrics.AveragePrecision(task="multilabel", num_labels=self.num_outputs, ignore_index=-100)
            self.test_aucroc = torchmetrics.AUROC(task="multilabel", num_labels=self.num_outputs, ignore_index=-100)
       
        elif self.task == 'multiclass':
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_outputs, ignore_index=-100)
            self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_outputs, average="macro", ignore_index=-100)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_outputs, ignore_index=-100)
            self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_outputs, average="macro", ignore_index=-100)

        elif self.task == 'regression':   
            self.val_r2 = torchmetrics.R2Score(num_outputs=2, multioutput='uniform_average')
            self.test_r2 = torchmetrics.R2Score(num_outputs=2, multioutput='uniform_average')
            self.val_arousal_r2 = torchmetrics.R2Score(num_outputs=1)
            self.test_arousal_r2 = torchmetrics.R2Score(num_outputs=1)
            self.val_valence_r2 = torchmetrics.R2Score(num_outputs=1)
            self.test_valence_r2 = torchmetrics.R2Score(num_outputs=1)


    @torch.no_grad()
    def update_and_log_metrics(self, split, y_pred, y):
        """
            update metrics per step
        """
        if self.task == 'regression':
            r2 = getattr(self, f"{split}_r2")
            r2.update(y_pred, y)
            self.log(f"{split}_r2", r2, on_epoch=True, prog_bar=True, sync_dist=True)
            arousal_r2 = getattr(self, f"{split}_arousal_r2")
            arousal_r2.update(y_pred[:, 0], y[:, 0])
            self.log(f"{split}_arousal_r2", arousal_r2, on_epoch=True, prog_bar=True, sync_dist=True)
            valence_r2 = getattr(self, f"{split}_valence_r2")
            valence_r2.update(y_pred[:, 1], y[:, 1])
            self.log(f"{split}_valence_r2", valence_r2, on_epoch=True, prog_bar=True, sync_dist=True)

        elif self.task == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=1)
            acc = getattr(self, f"{split}_acc")
            acc.update(y_pred, y)
            f1 = getattr(self, f"{split}_f1")
            f1.update(y_pred, y)
            self.log(f"{split}_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{split}_f1", f1, on_epoch=True, prog_bar=True, sync_dist=True)

        elif self.task == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
            ap = getattr(self, f"{split}_ap")
            ap.update(y_pred, y)
            auc = getattr(self, f"{split}_aucroc")
            auc.update(y_pred, y)
            self.log(f"{split}_ap", ap, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{split}_auc", auc, on_epoch=True, prog_bar=True, sync_dist=True)