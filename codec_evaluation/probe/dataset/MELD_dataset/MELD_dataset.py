import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from codec_evaluation.utils.utils import find_audios, cut_or_pad


class MELDdataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,        
        is_mono,
        is_normalize,
        audio_dir,
        meta_dir,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.is_normalize = is_normalize
        self.audio_dir = os.path.join(audio_dir, f'{split}')
        self.audio_file = None

        self.meta_dir = meta_dir
        self.meta_path = os.path.join(meta_dir, f'{split}_sent_emo.csv')
        
        self.metadata = pd.read_csv(self.meta_path, header=0)
        self.classes = """anger, disgust, fear, joy, neutral, sadness, surprise""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print(f"Error loading audio file {self.audio_file}: {e}")
            return None
        
    def get_item(self, index):
        """
        return:
            segments: [1,T]
            labels: [1]
        """
        dia_id = self.metadata.iloc[index].iloc[5]
        utt_id = self.metadata.iloc[index].iloc[6]
        self.audio_file = os.path.join(self.audio_dir, f'dia{dia_id}_utt{utt_id}.wav')
        
        segments, pad_mask= self.load_audio(self.audio_file)
        label = torch.tensor([self.class2id[self.metadata.iloc[index].iloc[3]]], dtype=torch.int64)
        segments = torch.vstack(segments)

        return {"audio": segments, "labels":  label, "n_segments": len(pad_mask)}

    def load_audio(
        self,
        audio_file,
    ):
        """
        input:
            audio_file:one of audio_file path
        return:
            waveform:[1,T]
        """
        if len(audio_file) == 0:
            raise FileNotFoundError("No audio files found in the specified directory.")

        try:
            waveform, _ = torchaudio.load(audio_file)
        except Exception as e:
            print(f"Error loading audio file {audio_file}: {e}")
            return None

        # Convert to mono if needed
        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize to [-1, 1] if needed
        if self.is_normalize:
            waveform = waveform / waveform.abs().max()

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length)

        return waveform, pad_mask

    def collate_fn(self, batch):
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]
        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.vstack(label_list).squeeze(1)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }


class MELDdataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_args, 
            codec_name,
            train_batch_size=32,
            valid_batch_size=2,
            test_batch_size=16,
            train_num_workers=8,
            valid_num_workers=4,
            test_num_workers=4):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MELDdataset(split="train", **self.dataset_args)
            self.valid_dataset = MELDdataset(split="valid", **self.dataset_args)
        if stage == "val":
            self.valid_dataset = MELDdataset(split="valid", **self.dataset_args)
        if stage == "test":
            self.test_dataset = MELDdataset(split="test", **self.dataset_args)

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