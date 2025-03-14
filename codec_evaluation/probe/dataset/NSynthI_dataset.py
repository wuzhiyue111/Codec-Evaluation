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
class NSynthIdataset(Dataset):
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
        self.audio_dir = audio_dir

        self.meta_dir = os.path.join(meta_dir, f'nsynth-{split}/examples.json')
        metadata = json.load(open(self.meta_dir,'r'))
        self.metadata = [(k, v['instrument_family_str']) for k, v in metadata.items()]
        self.audio_paths = [os.path.join(f'nsynth-{split}/audio', k+".wav") for k, v in self.metadata]
        self.class2id = {
            'bass': 0,
            'brass': 1,
            'flute': 2,
            'guitar': 3,
            'keyboard': 4,
            'mallet': 5,
            'organ': 6,
            'reed': 7,
            'string': 8,
            'synth_lead': 9,
            'vocal': 10
        }
        self.id2class = {v: k for k, v in self.class2id.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.getitem(index)

    
    def getitem(self, index):
        """
            return:
                segments: [n_segments, segments_length]
                labels: [n_segments, 50]
        """
        audio_path = self.audio_paths[index]
        audio_file = os.path.join(self.audio_dir, audio_path)
        segments, pad_mask = self.load_audio(audio_file)
        class_name = self.metadata[index][1]
        label = torch.tensor(self.class2id[class_name])

        labels = []
        for mask in pad_mask:
            if mask == 1:
                labels.append(label)
            else:
                labels.append(-100)
        labels = torch.tensor(labels)
        segments = torch.vstack(segments)

        return {"audio": segments, "labels":  labels, "n_segments": len(pad_mask)}
    
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

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length)

        return waveform, pad_mask
    
    def collate_fn(self, batch):
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]

        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.cat(label_list,dim=0)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }


class NSynthIdataModule(pl.LightningDataModule):
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
            self.train_dataset = NSynthIdataset(split="train", **self.dataset_args)
            # 创建验证集
            self.valid_dataset = NSynthIdataset(split="valid", **self.dataset_args)
        if stage == "val":
            self.valid_dataset = NSynthIdataset(split="valid", **self.dataset_args)
        if stage == "test":
            self.test_dataset = NSynthIdataset(split="test", **self.dataset_args)
 
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