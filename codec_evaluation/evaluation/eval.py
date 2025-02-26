import torch
import sys  
import os
import torchmetrics 
import argparse
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from einops import reduce
sys.path.append(os.path.expanduser('~/project/Codec-Evaluation'))
from codec_evaluation.evaluation.dataset.EMO_dataset import EMOdataset
from codec_evaluation.evaluation.utils import init_codec, cut_or_pad


parser = argparse.ArgumentParser(description="Audio processing parameters")

parser.add_argument('--is_mono', type=bool, default=True, help='Convert audio to mono')
parser.add_argument('--is_normalize', type=bool, default=False, help='Normalize audio to [-1, 1]')
parser.add_argument('--audio_dir', type=str, default='/home/wsy/project/MARBLE-Benchmark/data/EMO/emomusic/wav')
parser.add_argument('--meta_dir', type=str, default='/home/wsy/project/MARBLE-Benchmark/data/EMO/emomusic')
parser.add_argument('--codecname', type=str, default='encodec')
parser.add_argument('--mode', type=str, default="unquantized_emb")
parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the audio')
parser.add_argument('--target_sec', type=int, default=30 ,help='Target audio length in seconds')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
parser.add_argument('--n_segments', type=int, default=6, help='Number of segments per audio')

args = parser.parse_args()

class EMOProber(pl.LightningModule):
    """
        self.num_outputs: Number of label categories;
        self.codec: Initialied codec;
        self.token_rate: The number of tokens per sec;
        self.in_ch: The number of channels for deep convolution;
        self.dim: The D dimension of the codec output;
        
        metrics:
            r2、arousal_r2、 valence_r2
    """
    def __init__(self, device):
        super(EMOProber, self).__init__()
        self.num_outputs = 2  
        self.codec = init_codec(modelname = args.codecname,      
                                sample_rate = args.sample_rate, 
                                mode = args.mode, 
                                device = device, 
                                freeze = True)
        
        self.token_rate = self.codec.token_rate
        self.in_ch = self.dim = self.codec.dim  

        self.channel_attention = nn.Sequential(
            SEBlock(channel = self.dim, reduction = 16),
            SEBlock(channel = self.dim, reduction = 16),
            SEBlock(channel = self.dim, reduction = 16)
            )

        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dsconv = DSConv(in_ch=self.in_ch,                
                             out_ch=self.in_ch , 
                             kernel_size=self.kernel_size, 
                             stride=self.stride, 
                             padding=self.padding)
        
        self.init_linear()
        self.init_metrics()  
        self.to(device)
    
    def init_linear(self):
        self.linear = nn.Linear(self.dim, self.dim // 16)

        self.target_T = self.token_rate * args.target_sec // args.n_segments
        T1 = math.floor((self.target_T + 2 * self.padding - self.kernel_size) / self.stride)  + 1 
        input_dim = (math.floor((T1 + 2 * self.padding - self.kernel_size) / self.stride) + 1) * self.dim // 16

        self.output = nn.Linear(input_dim, self.num_outputs)


    def extract_feature(self, waveforms):
        all_features = []

        for i in range(waveforms.shape[0]):
            waveform = waveforms[i].unsqueeze(0) # [1, T]
            length = torch.tensor([1.])
            features = self.codec(waveform, length)
            
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0) 

        all_features = cut_or_pad(waveform=all_features, 
                                  target_length=self.target_T, 
                                  codecname=args.codecname ) 

        if args.codecname == 'semanticodec':
            if all_features.dim() == 4:
                split_feature = all_features.unbind(dim=2) # [B*n_segments, D, 2, T]
                
            else:
                split_feature = all_features.chunk(2, dim=1) # [tensor[B*n_segments, D, T]、tensor[B*n_segments, D, T]]

            all_features = torch.cat(split_feature, dim=0)  # [2*B*n_segments, D, T]
        
        return all_features


    def forward(self, x):
        x = x.float()  #[B*n_segments, D, T] 

        x_channel = self.channel_attention(x)
        x_conv = self.dsconv(x_channel)  
        x_channel = self.channel_attention(x_conv)
        x_conv = self.dsconv(x_channel)    #[B*n_segments, D, T']

        x_conv = x_conv.permute(0, 2, 1)  #[B*n_segments, T', D]
        x_conv = self.linear(x_conv)      #[B*n_segments, T', D//16]

        x_flattened = x_conv.flatten(start_dim=1, end_dim=-1)    #[B*n_segments, input_dim=T' * D//16]

        output = self.output(x_flattened)  #[B*n_segments, 2]

        return output   

    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        """Training step."""
        x, y = batch

        x = self.extract_feature(x)
        # [a,s] = x.chunk(2,dim=0)
        # y_pred = self(a)

        y_pred = self(x)  ##[B*n_segments, 2]

        loss = F.mse_loss(y_pred, y) 
        # loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Gradient: {param.grad.max()}")
        # import pdb; pdb.set_trace()
        self.log('train_loss', loss, batch_size=args.batch_size, on_epoch=True, prog_bar=True, logger=True)
        self.update_metrics("train", y, y_pred)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        x = self.extract_feature(x) 
        # [a,s] = x.chunk(2,dim=0)
        # y_pred = self(a)
        y_pred = self(x)
        y_pred = reduce(y_pred, '(b g) n -> b n', reduction = 'mean', g = args.n_segments) 
            
        loss = F.mse_loss(y_pred, y)
        self.log('valid_loss', loss, batch_size=args.batch_size, on_epoch=True, prog_bar=True, logger=True)
        self.update_metrics("valid", y, y_pred)

        return loss


    def configure_optimizers(self):
        """Configure the optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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

    def training_epoch_end(self, outputs):
      
        self.log_metrics('train')
    
    def validation_epoch_end(self, outputs):
        self.log_metrics('valid')
    

class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, x):
        """
        input: [B, T, D]
        B: batch size
        T: 时间维度
        D: 特征维度（即音频的通道数）
        """
        b, d, t = x.shape  
        y = self.avg_pool(x)    
    
        y = y.view(b, d)  
        y = self.fc(y)    
        
        y = y.view(b, d, 1)  
        return x * y.expand_as(x)  
    

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(DSConv, self).__init__()
        # deep convolution
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,  
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch, 
            bias=False
        )

    def forward(self, x):
        x = self.depthwise_conv(x) 
        
        return x


class EMOdataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size, val_split, device):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.device = device
        self.train_dataset = None
        self.valid_dataset = None
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_size = int((1 - self.val_split) * len(self.dataset))
            valid_size = len(self.dataset) - train_size
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size])    


    def train_collate_fn(self, batch):
        """
            return:
                features_tensor:(batch_size * n_segments, length)
                labels_tensor:(batch_size * n_segments, 2)
                if codecname='semanticodec'
                    labels_tensor:(batch_size * n_segments * 2, 2)
        """
        features, labels = zip(*batch)  
        features_tensor = torch.cat(features, dim=0)  
        labels_tensor = torch.cat(labels, dim=0)      
        # import pdb;pdb.set_trace()
        if args.codecname == 'semanticodec':
            labels_tensor = torch.cat([labels_tensor,labels_tensor], dim = 0)

        return features_tensor, labels_tensor
    
    def valid_collate_fn(self, batch):
        """
            return:
                features_tensor:(batch_size * n_segments, length)
                labels_tensor:(batch_size , 2)
                if codecname='semanticodec'
                    labels_tensor:(batch_size * 2, 2)
        """
        
        features, labels = zip(*batch)  

        features_tensor = torch.cat(features, dim=0) 
        labels_tensor = torch.cat(labels, dim=0)   
        labels_tensor = reduce(labels_tensor, '(b g) n -> b n', reduction = 'mean', g = args.n_segments)  

        if args.codecname == 'semanticodec':
            labels_tensor = torch.cat([labels_tensor,labels_tensor], dim = 0)

        return features_tensor, labels_tensor
    

    def train_dataloader(self):  #pytorch_lightning-1.9.4
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, collate_fn=self.train_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.batch_size, shuffle=False, collate_fn=self.valid_collate_fn)
    

def callback():
    checkpoint_callback = ModelCheckpoint(
            monitor='valid_loss',  
            dirpath='./checkpoints', 
            filename= args.codecname +args.mode+ '_best_model',  
            save_top_k=1,  
            mode='min',  
            save_weights_only=True,  
        )
    return [checkpoint_callback]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = EMOdataset(sample_rate=args.sample_rate, 
                         target_sec=args.target_sec, 
                         n_segments=args.n_segments, 
                         is_mono=args.is_mono, 
                         is_normalize=args.is_normalize, 
                         audio_dir=args.audio_dir, 
                         meta_dir=args.meta_dir, 
                         device=device)

    dataloader = EMOdataModule(dataset=dataset, 
                               batch_size=args.batch_size, 
                               val_split=args.val_split, 
                               device=device)
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    valid_dataloader = dataloader.val_dataloader()
    model = EMOProber(device).to(device)

    trainer = pl.Trainer(accelerator='gpu' if device == 'cuda' else 'cpu', 
                                              devices=1 if device == 'cuda' else 0, 
                                              max_epochs=args.max_epochs,
                                              callbacks=callback(),
                                              log_every_n_steps=50,
                                              accumulate_grad_batches=20,
                                              )
    
    # Train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()