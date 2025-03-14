import torch
import torchmetrics 
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from codec_evaluation.init_codecs import init_codec
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
                 probe_model_builder: Any = None,
                 optimizer_builder: Any = None,
                 lr_scheduler_builder: Any = None,
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
                                freeze = True)
        self.codec_name = codec_name
        self.sample_rate = sample_rate
        
        if codec_name == "semanticodec":
            self.dim = self.codec.dim * 2
            self.token_rate = self.codec.token_rate /2
        else:
            self.dim = self.codec.dim  
            self.token_rate = self.codec.token_rate

        self.target_T = int(self.token_rate * target_sec )
        self.audio_length = target_sec * self.sample_rate
        self.probe_model = probe_model_builder(
            codec_dim = self.dim,
            target_T = self.target_T)
        
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
            return: [B*n_segments, D, T]
        """
        length = torch.ones(waveforms.shape[0])
        all_features = self.codec(waveforms, length)
        # import pdb; pdb.set_trace()
        if self.codec_name == 'semanticodec' or self.codec_name == 'mimi':
            assert expect_lenth is not None, "expect_lenth is required for semanticodec"
            if all_features.dim() == 4:
                all_features = rearrange(all_features, 'b d c t -> b (d c) t')
            max_length = expect_lenth
            all_features = all_features[:, :, :int(max_length)]

        return all_features

        # consider the resample function of codec, libriTTS dataset is 24000 sample rate

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, batch_size, labels_pred, labels = self.step(batch)

        self.log('train_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_metrics("train", labels, labels_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, batch_size, labels_pred, labels = self.step(batch)
        self.log('valid_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_metrics("valid", labels, labels_pred)

        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        
        loss, batch_size, labels_pred, labels = self.step(batch)

        self.log('test_loss', loss, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        self.update_metrics("test", labels, labels_pred)
    
    def step(self, batch):
        audio = batch["audio"]
        labels = batch["labels"]
        n_segments_list = batch["n_segments_list"]

        batch_size = labels.shape[0]
        if self.codec.orig_sample_rate != self.sample_rate:
            feature_length = self.audio_length * (self.codec.orig_sample_rate / self.sample_rate) // self.codec.hop_length
        else:
            feature_length = self.audio_length // self.codec.hop_length

        audio_features = self.extract_feature(audio, int(feature_length))

        loss, labels_pred = self.probe_model(audio_features, labels, n_segments_list)
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
        self.all_metrics = set()
        if self.task == 'multilabel':
            for split in ['train', 'valid', 'test']:
                setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                                                            task=self.task,
                                                            num_labels=self.num_outputs,
                                                            ignore_index=-100))
                self.all_metrics.add('ap')

                setattr(self, f"{split}_aucroc", torchmetrics.AUROC(
                                                            task=self.task,
                                                            num_labels=self.num_outputs,
                                                            ignore_index=-100))
                self.all_metrics.add('aucroc')
       

        elif self.task == 'multiclass':
            for split in ['train', 'valid', 'test']:
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                                                            task=self.task,
                                                            num_classes=self.num_outputs,
                                                            ignore_index=-100))
                self.all_metrics.add('acc')
                setattr(self, f"{split}_f1", torchmetrics.F1Score(
                                                                    task=self.task,
                                                                    num_classes=self.num_outputs,
                                                                    average='macro',
                                                                    ignore_index=-100))
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

    def save_result(self):
        if self.task == 'regression':
            r2 = getattr(self, f"test_r2").compute()
            arousal_r2 = getattr(self, f"test_arousal_r2").compute()
            valence_r2 = getattr(self, f"test_valence_r2").compute()
            self.test_step_outputs.append({"arousal_r2": arousal_r2, "valence_r2": valence_r2})

            getattr(self, f"test_valence_r2").reset()
            getattr(self, f"test_r2").reset()
            getattr(self, f"test_arousal_r2").reset()
        elif self.task == 'multiclass':
            acc = getattr(self, f"test_acc").compute()
            f1 = getattr(self, f"test_f1").compute()
            self.test_step_outputs.append({"acc": acc, "f1": f1})
            getattr(self, f"test_f1").reset()
            getattr(self, f"test_acc").reset()
        elif self.task == 'multilabel':
            ap = getattr(self, f"test_ap").compute()
            aucroc = getattr(self, f"test_aucroc").compute()
            self.test_step_outputs.append({"ap": ap, "aucroc": aucroc})

            getattr(self, f"test_aucroc").reset()
            getattr(self, f"test_ap").reset()
    def on_train_epoch_end(self, outputs = None):
        self.log_metrics('train')
    
    def on_validation_epoch_end(self, outputs = None):        
        self.log_metrics('valid')

    def on_test_epoch_end(self, outputs = None):
        self.save_result()
        list0 = []
        list1 = []

        if self.task == 'regression':
            for output in self.test_step_outputs:
                list0.append(output["arousal_r2"])
                list1.append(output["valence_r2"])
            avg_0 = sum(list0) / len(list0)
            avg_1 = sum(list1) / len(list1)
            self.test_step_outputs = {"arousal_r2": avg_0, "valence_r2": avg_1}
        elif self.task == 'multiclass':
            for output in self.test_step_outputs:
                list0.append(output["acc"])
                list1.append(output["f1"])
            avg_0 = sum(list0) / len(list0)
            avg_1 = sum(list1) / len(list1)
            
            self.test_step_outputs = {"acc": avg_0, "f1": avg_1}
        elif self.task == 'multilabel':
            for output in self.test_step_outputs:
                list0.append(output["ap"])
                list1.append(output["aucroc"])
            avg_0 = sum(list0) / len(list0)
            avg_1 = sum(list1) / len(list1)
            
            self.test_step_outputs = {"ap": avg_0, "aucroc": avg_1}