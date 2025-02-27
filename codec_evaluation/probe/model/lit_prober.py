import torch
import torchmetrics 
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import reduce

from codec_evaluation.utils.utils import init_codec, cut_or_pad
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
                 mode: str = 'quantized_emb',
                 target_sec: int = 30,
                 n_segments: int = 6,
                 loss_type: str = 'mse',
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
        self.num_outputs = 2  
        self.codec = init_codec(modelname = codec_name,      
                                sample_rate = sample_rate, 
                                mode = mode, 
                                device = 'cpu', 
                                freeze = True)

        self.codec_name = codec_name
        self.loss_type = loss_type
        self.token_rate = self.codec.token_rate
        self.in_ch = self.dim = self.codec.dim  
        self.target_T = self.token_rate * target_sec // n_segments
        self.n_segments = n_segments
        self.probe_model = probe_model_builder(
            codec_dim = self.dim,
            token_rate = self.token_rate)

        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.init_metrics()  

    def extract_feature(self, waveforms):
        """
            extract features from codec
            waveforms: [B, T]
            return: [B*n_segments, D, T]
        """
        all_features = []

        for i in range(waveforms.shape[0]):
            waveform = waveforms[i].unsqueeze(0) # [1, T]
            length = torch.tensor([1.])
            features = self.codec(waveform, length)

            all_features.append(features)

        all_features = torch.cat(all_features, dim=0) 

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

        y_pred = self.probe_model(x)

        if self.loss_type == 'mse':
            loss = F.mse_loss(y_pred, y)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(y_pred, y)

        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        self.update_metrics("train", y, y_pred)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        batch_size = y.shape[0]
        x = self.extract_feature(x) 

        y_pred = self.probe_model(x)
        y_pred = reduce(y_pred, '(b g) n -> b n', reduction = 'mean', g = self.n_segments) 
            
        loss = F.mse_loss(y_pred, y)
        self.log('valid_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        self.update_metrics("valid", y, y_pred)

        return loss


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
        for split in ['train', 'valid']:
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
        getattr(self, f"{split}_r2").update(y_pred, y)
        getattr(self, f"{split}_arousal_r2").update(y_pred[:, 0], y[:, 0])
        getattr(self, f"{split}_valence_r2").update(y_pred[:, 1], y[:, 1])

    @torch.no_grad()
    def log_metrics(self, split):
        """
            log metrics at the end of epoch
        """
        self.log(f"{split}_r2", getattr(self, f"{split}_r2").compute(), sync_dist=True)
        getattr(self, f"{split}_r2").reset()
        self.log(f"{split}_arousal_r2", getattr(self, f"{split}_arousal_r2").compute(), sync_dist=True)
        getattr(self, f"{split}_arousal_r2").reset()
        self.log(f"{split}_valence_r2", getattr(self, f"{split}_valence_r2").compute(), sync_dist=True)
        getattr(self, f"{split}_valence_r2").reset()

    def on_train_epoch_end(self, outputs):
        self.log_metrics('train')
    
    def on_validation_epoch_end(self, outputs):
        self.log_metrics('valid')