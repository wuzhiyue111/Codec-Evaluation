import torch
import torchmetrics 
import torch.nn.functional as F
import pytorch_lightning as pl
from codec_evaluation.init_codecs import init_codec
from codec_evaluation.utils.utils import cut_or_pad
from typing import Any

class Prober(pl.LightningModule):
    """
        self.num_outputs: Number of label categories;
        self.codec: Initialied codec;
        self.token_rate: The number of tokens per sec;
        self.in_ch: The number of channels for deep convolution;
        self.dim: The D dimension of the codec output;
        
        metrics:
            r2、arousal_r2、 valence_r2
    """
    def __init__(self, 
                 codec_name: str,
                 sample_rate: int,
                 model_ckpt_dir: str,
                 task: str,
                 num_outputs: int,
                 mode: str = 'quantized_emb',
                 target_sec: int = 30,
                 n_segments: int = 6,
                 probe_model_builder: Any = None,
                 optimizer_builder: Any = None,
                 lr_scheduler_builder: Any = None,
                 ):
        """
            codec_name must in ['dac', 'encodec', 'mimi', 'semanticodec', 'speechtokenizer', wavtokenizer, wavlm_kmeans]
            sample_rate: the audio sample_rate when you are training the probe model
            mode must in ['quantized_emb', 'unquantized_emb']
        """
        super(Prober, self).__init__()  
        self.codec = init_codec(modelname = codec_name,      
                                sample_rate = sample_rate, 
                                mode = mode, 
                                model_ckpt_dir = model_ckpt_dir,
                                device = 'cpu', 
                                freeze = True)
        self.codec_name = codec_name
        self.token_rate = self.codec.token_rate
        self.in_ch = self.dim = self.codec.dim  
        self.target_T = int(self.token_rate * target_sec // n_segments)
        self.n_segments = n_segments
        self.probe_model = probe_model_builder(
            codec_dim = self.dim,
            target_T = self.target_T)
        
        self.num_outputs = num_outputs
        self.task = task
        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.init_metrics()  
    

    def extract_feature(self, waveforms):
        """
            extract features from codec
            waveforms: [B, T]
            return: [B*n_segments, D, T]
        """
        length = torch.ones(waveforms.shape[0])
        all_features = self.codec(waveforms, length)

        all_features = cut_or_pad(waveform=all_features, 
                                  target_length=self.target_T, 
                                  codecname=self.codec_name) 

        if self.codec_name == 'semanticodec':
            if all_features.dim() == 4:
                split_feature = all_features.unbind(dim=2) # [B*n_segments, D, 2, T]
            else:
                split_feature = all_features.chunk(2, dim=1) # [tensor[B*n_segments, D, T]、tensor[B*n_segments, D, T]]

            all_features = torch.cat(split_feature, dim=0)  # [2*B*n_segments, D, T]

        return all_features


    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        batch_size = y.shape[0]

        x = self.extract_feature(x)

        loss, y_pred = self.probe_model(x, y)

        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_metrics("train", y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        batch_size = y.shape[0]
        x = self.extract_feature(x) 

        loss, y_pred = self.probe_model(x, y)

        self.log('valid_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_metrics("valid", y, y_pred)

        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        batch_size = y.shape[0]
        x = self.extract_feature(x) 

        loss, y_pred = self.probe_model(x, y)

        self.log('test_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_metrics("test", y, y_pred)


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
        self.all_metrics = set()
        if self.task == 'multilabel':
            for split in ['train', 'valid', 'test']:
                setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                                                            task=self.task,
                                                            num_labels=self.num_outputs))
                self.all_metrics.add('ap')

                setattr(self, f"{split}_aucroc", torchmetrics.AUROC(
                                                            task=self.task,
                                                            num_labels=self.num_outputs))
                self.all_metrics.add('aucroc')
       

        elif self.task == 'multiclass':
            for split in ['train', 'valid', 'test']:
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                                                            task=self.task,
                                                            num_classes=self.num_outputs))
                self.all_metrics.add('acc')
                setattr(self, f"{split}_f1", torchmetrics.F1Score(
                                                                    task=self.task,
                                                                    num_classes=self.num_outputs,
                                                                    average='macro'))
                self.all_metrics.add('f1')

        elif self.task == 'regression':        
            for split in ['train', 'valid', 'test']:
                # r2 score
                setattr(self, f"{split}_r2", torchmetrics.R2Score(num_outputs=2, multioutput='uniform_average'))
                self.all_metrics.add('r2')
                setattr(self, f"{split}_arousal_r2", torchmetrics.R2Score(num_outputs=1))
                self.all_metrics.add('arousal_r2')
                setattr(self, f"{split}_valence_r2", torchmetrics.R2Score(num_outputs=1))
                self.all_metrics.add('valence_r2')

    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        """
            update metrics per step
        """
        if self.task == 'regression':
            getattr(self, f"{split}_r2").update(y_pred, y)
            getattr(self, f"{split}_arousal_r2").update(y_pred[:, 0], y[:, 0])
            getattr(self, f"{split}_valence_r2").update(y_pred[:, 1], y[:, 1])
        elif self.task == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=1)
            getattr(self, f"{split}_acc").update(y_pred, y)
            getattr(self, f"{split}_f1").update(y_pred, y)
        elif self.task == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
            getattr(self, f"{split}_ap").update(y_pred, y)
            getattr(self, f"{split}_aucroc").update(y_pred, y)

    @torch.no_grad()
    def log_metrics(self, split):
        """
            log metrics at the end of epoch
        """
        if self.task == 'regression':
            self.log(f"{split}_r2", getattr(self, f"{split}_r2").compute(), sync_dist=True)
            getattr(self, f"{split}_r2").reset()
            self.log(f"{split}_arousal_r2", getattr(self, f"{split}_arousal_r2").compute(), sync_dist=True)
            getattr(self, f"{split}_arousal_r2").reset()
            self.log(f"{split}_valence_r2", getattr(self, f"{split}_valence_r2").compute(), sync_dist=True)
            getattr(self, f"{split}_valence_r2").reset()
        elif self.task == 'multiclass':
            self.log(f"{split}_acc", getattr(self, f"{split}_acc").compute(), sync_dist=True)
            getattr(self, f"{split}_acc").reset()
            self.log(f"{split}_f1", getattr(self, f"{split}_f1").compute(), sync_dist=True)
            getattr(self, f"{split}_f1").reset()

        elif self.task == 'multilabel':
            self.log(f"{split}_ap", getattr(self, f"{split}_ap").compute(), sync_dist=True)
            getattr(self, f"{split}_ap").reset()
            self.log(f"{split}_aucroc", getattr(self, f"{split}_aucroc").compute(), sync_dist=True)
            getattr(self, f"{split}_aucroc").reset()
            

    def on_train_epoch_end(self, outputs = None):
        self.log_metrics('train')
    
    def on_validation_epoch_end(self, outputs = None):
        self.log_metrics('valid')

    def on_test_epoch_end(self, outputs = None):
        self.log_metrics('test')
