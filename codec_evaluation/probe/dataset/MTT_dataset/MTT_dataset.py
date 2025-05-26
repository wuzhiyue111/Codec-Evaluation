import os
import torchaudio
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_from_disk
from codec_evaluation.utils.utils import cut_or_pad
import numpy as np
from codec_evaluation.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

class MTTdataset(Dataset):
    def __init__(
        self,
        sample_rate,
        target_sec,        
        is_mono,
        dataset_path,
        base_audio_dir,  
        task
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.dataset_path = dataset_path
        self.base_audio_dir = base_audio_dir  
        self.task = task
        
        self.dataset = load_from_disk(dataset_path)
        
        self.uuids = self.dataset["uuid"]
        self.audio_paths = self.dataset["audio_path"]
        
        self.labels = np.load(os.path.join(base_audio_dir, 'MTT', 'binary_label.npy'))

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            audio_path = self.dataset[index]["audio_path"]  # 数据集中的相对路径
            full_path = os.path.join(self.base_audio_dir, audio_path)
            logger.error(f"Error loading {full_path}: {e}")
            return None  
    
    def getitem(self, index):
        """
            return:
                segments: [n_segments, segments_length]
                labels: [n_segments, 50]
        """
        uuid = self.uuids[index]
        audio_path = self.audio_paths[index]
        audio_file = os.path.join(self.base_audio_dir, audio_path) 
        segments, pad_mask = self.load_audio(audio_file)
        label = torch.from_numpy(self.labels[uuid])
        segments = torch.vstack(segments)

        return {"audio": segments, "labels":  label, "n_segments": len(pad_mask)}
    
    def load_audio(
        self, 
        audio_file
    ):
        """
        input:
            audio_file:one of audio_file path
        return:
            waveform:[n,T]
        """
        waveform, _ = torchaudio.load(audio_file)

        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length, task = self.task)

        return waveform, pad_mask 


    def collate_fn(self, batch):
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]

        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.vstack(label_list).long()

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }


class MTTdataModule(pl.LightningDataModule):
    def __init__(self,
            dataset_args, 
            train_audio_dir,
            valid_audio_dir,
            test_audio_dir,
            codec_name,
            train_batch_size=32,
            valid_batch_size=2,
            test_batch_size=16,
            train_num_workers=8,
            valid_num_workers=4,
            test_num_workers=4
        ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_audio_dir = train_audio_dir
        self.valid_audio_dir = valid_audio_dir
        self.test_audio_dir = test_audio_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MTTdataset(dataset_path=self.test_audio_dir, **self.dataset_args)
            self.valid_dataset = MTTdataset(dataset_path=self.valid_audio_dir, **self.dataset_args) 
        if stage == "val":
            self.valid_dataset = MTTdataset(dataset_path=self.valid_audio_dir, **self.dataset_args)
        if stage == "test":
            self.test_dataset = MTTdataset(dataset_path=self.test_audio_dir, **self.dataset_args)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            collate_fn=self.valid_dataset.collate_fn,
            num_workers=self.valid_num_workers,
        )
    
    def test_dataloader(self) :
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )