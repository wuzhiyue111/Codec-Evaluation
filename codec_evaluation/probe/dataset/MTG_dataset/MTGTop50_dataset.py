import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import reduce

from codec_evaluation.utils.utils import cut_or_pad


class MTGTop50dataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,        
        is_mono,
        is_normalize,
        audio_dir,
        meta_dir,
        task,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.is_normalize = is_normalize
        self.audio_dir = audio_dir
        self.task = task

        self.meta_dir = os.path.join(meta_dir, f'data/splits/split-0/autotagging_top50tags-{split}.tsv')
        self.metadata = open(self.meta_dir, 'r').readlines()[1:]
        self.all_paths = [line.split('\t')[3] for line in self.metadata]
        self.all_tags = [line.split('\t')[5:] for line in self.metadata]

        self.class2id = self.read_class2id(meta_dir)
        self.id2class = {v: k for k, v in self.class2id.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        """
        return:
            segments: [1, wavform_length]
            labels: [1, 20]
        """
        audio_path = self.all_paths[index]
        audio_path = audio_path.replace('.mp3', '.low.mp3')
        
        audio_file = os.path.join(self.audio_dir, audio_path)
        segments, pad_mask = self.load_audio(audio_file)
        class_name = self.all_tags[index]
        label = torch.zeros(len(self.class2id))
        for c in class_name:
            label[self.class2id[c.strip()]] = 1

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
            waveform:[T]
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
        
        if waveform.shape[1] > 150 *self.sample_rate:
            waveform = waveform[:, :150 *self.sample_rate]

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length, task = self.task)

        return waveform, pad_mask 

    def read_class2id(self, meta_dir):
        class2id = {}
        for split in ['train', 'validation']:
            data = open(os.path.join(meta_dir, f'data/splits/split-0/autotagging_top50tags-{split}.tsv'), "r").readlines()
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id:
                        class2id[tag] = len(class2id)
        return class2id

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


class MTGTop50Module(pl.LightningDataModule):
    def __init__(self,
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
            self.train_dataset = MTGTop50dataset(split="train", **self.dataset_args)
            self.valid_dataset = MTGTop50dataset(split="validation", **self.dataset_args)
        if stage == "val":
            self.valid_dataset = MTGTop50dataset(split="validation", **self.dataset_args)
        if stage == "test":
            self.test_dataset = MTGTop50dataset(split="test", **self.dataset_args)

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