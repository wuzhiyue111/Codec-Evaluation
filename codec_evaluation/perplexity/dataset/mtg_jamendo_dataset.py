import os
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_from_disk
from codec_evaluation.utils.logger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
import numpy as np


logger = RankedLogger(__name__, rank_zero_only=True)

class Jamendo_ppl_dataset(Dataset):
    def __init__(
        self,
        base_audio_dir,  # audio file root directory (used for splicing path)
        dataset_path,    # .arrow file path
        split,
        sample_rate,
        target_sec,
        is_mono=True,       
    ):
        super().__init__()
        dataset = load_from_disk(dataset_path)  
        if split is not None:
            dataset = dataset.filter(lambda x: x['split'] == split)  
        self.dataset = dataset
        self.base_audio_dir = base_audio_dir
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = sample_rate * target_sec  
        self.is_mono = is_mono
        print(f"Found {len(self.dataset)} audio files in {base_audio_dir}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            audio_file = self.dataset[index]['audio_path']
            audio_path = os.path.join(self.base_audio_dir, audio_file)
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return None  

    def get_item(self, index):
        sample = self.dataset[index]
        audio_path = os.path.join(self.base_audio_dir, sample["audio_path"])
        
        waveform, _ = torchaudio.load(audio_path)
        if self.is_mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  
        
        # Get the original audio length (in this case, orig_length >= 30 seconds)
        orig_length = waveform.shape[1]
        
        # Randomly extract 15 seconds
        max_start = orig_length - self.target_length
        start = random.randint(0, max_start)
        end = start + self.target_length
        waveform = waveform[:, start:end]  
        audio_length = self.target_length  
        
        return {
            "audio": waveform,           
            "audio_length": audio_length,  
        }

    def collate_fn(self, batch):
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]

        audio_tensor = pad_sequence(audio_list, batch_first=True)
        audio_length_tensor = torch.tensor(audio_length_list)
        
        return {
            "audio": audio_tensor,
            "audio_length": audio_length_tensor,
        }
    
class Jamendo_ppl_Module(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        base_audio_dir=None,
        sample_rate=44100,
        target_sec=15,  # Target segment length (seconds)
        is_mono=True,
        train_batch_size=16,
        valid_batch_size=16,
        test_batch_size=16,
        train_num_workers=4,
        valid_num_workers=4,
        test_num_workers=4,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.dataset_path = dataset_path
        self.base_audio_dir = base_audio_dir
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.is_mono = is_mono

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Jamendo_ppl_dataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.dataset_path,
                split="train",
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)
            self.valid_dataset = Jamendo_ppl_dataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.dataset_path,
                split="validation",
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)
        if stage == "val":
            self.valid_dataset = Jamendo_ppl_dataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.dataset_path,
                split="validation",
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)
        if stage == "test":
            self.test_dataset = Jamendo_ppl_dataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.dataset_path,
                split="test",
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )