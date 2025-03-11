import os
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torchaudio
import torch

import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from codec_evaluation.utils.utils import find_audios, cut_or_pad
from einops import reduce
import numpy as np
import json
class NSynthPdataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,
        n_segments,
        is_mono,
        is_normalize,
        audio_dir,
        meta_dir,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.n_segments = n_segments
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.is_normalize = is_normalize
        self.audio_dir = audio_dir

        self.meta_dir = os.path.join(meta_dir, f'nsynth-{split}/examples.json')
        metadata = json.load(open(self.metadata_dir,'r'))
        self.metadata = [(k, v['pitch']) for k, v in metadata.items()]
        self.audio_paths = [os.path.join(f'nsynth-{split}/audio', k+".wav") for k, v in self.metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            return None
    
    def getitem(self, index):
        """
            return:
                segments: [n_segments, segments_length]
                labels: [n_segments, 50]
        """
        audio_path = self.audio_paths[index]
        audio_file = os.path.join(self.audio_dir, audio_path)
        waveform = self.load_audio(audio_file)
        pitch = self.metadata[index][1]  # actually the pitch range is [9, 120]
        label = pitch - 1

        segments = self.split_audio(waveform, self.n_segments)
        segments = torch.stack(segments, dim=0)

        labels = label.unsqueeze(0).repeat(self.n_segments, 1)

        return segments, labels
    
    def load_audio(
            self,
            audio_file,
    ):
        """
        input:
            audio_file:one of audio_file path
        return:
            waveform:[T]
        """
        if len(audio_file) == 0:
            raise FileNotFoundError("No audio files found in the specified directory.")

        try:
            waveform, _ = torchaudio.load(audio_file)
        except Exception as e:
            print(f"Error loading audio file {audio_file}:{e}")
            return None
        
        # Convert to mono if needed
        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize to [-1, 1] if needed
        if self.is_normalize:
            waveform = waveform / waveform.abs().max()

        waveform = cut_or_pad(waveform=waveform, target_length=self.target_length)
        waveform = waveform.squeeze(0)

        return waveform
    

    def split_audio(self, waveform, n_segments):
        """
        input:
            waveform:[T];
            n_segments: the number of segments per audio
        return:
            segments:[n_segments,segment_length]
        """
        segment_length = self.target_length // n_segments   # length of per segment
        segments = []

        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = waveform[start:end]
            segments.append(segment)

        return segments


class NSynthPdataModule(pl.LightningDataModule):
    def __init__(self, dataset_args, batch_size, codec_name, num_workers):
        super().__init__()
        self.dataset_args = dataset_args
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.codec_name = codec_name
        self.n_segments = dataset_args["n_segments"]
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = NSynthPdataset(split="train", **self.dataset_args)
            # 创建验证集
            self.valid_dataset = NSynthPdataset(split="valid", **self.dataset_args) 
    
    def train_collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size * n_segments, length)
            labels_tensor:(batch_size * n_segments, 50)
            if codecname='semanticodec'
                labels_tensor:(batch_size * n_segments * 2, 50)
        """
        batch = [b for b in batch if b is not None]  # 过滤掉 None
        if len(batch) == 0:
            return None
        features, labels = zip(*batch)
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        labels_tensor = labels_tensor.squeeze(1)  #(B,1)

        if self.codec_name == "semanticodec":
            labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)

        return features_tensor, labels_tensor
    
    def valid_collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size * n_segments, length)
            labels_tensor:(batch_size , 128)
            if codecname='semanticodec'
                labels_tensor:(batch_size * 2, 128)
        """
        features, labels = zip(*batch)
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        labels_tensor = labels_tensor.squeeze(1)

        if self.codec_name == "semanticodec":
            labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)

        return features_tensor, labels_tensor
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=self.valid_collate_fn,
            num_workers=self.num_workers,
        )