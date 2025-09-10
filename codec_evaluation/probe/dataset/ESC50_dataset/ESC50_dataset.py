import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from codec_evaluation.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

class ESC50dataset(Dataset):
    def __init__(
        self,
        sample_rate,
        target_sec,
        is_mono,
        dataset_path, 
        task,
        base_audio_dir, 
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.is_mono = is_mono
        self.task = task
        self.base_audio_dir = base_audio_dir

        self.dataset = load_from_disk(dataset_path)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            audio_path = self.dataset[index]["audio_path"]
            full_path = os.path.join(self.base_audio_dir, audio_path)
            logger.error(f"Error loading audio file {full_path}: {e}")
            return None
        
    def get_item(self, index):
        """
        return:
            audio: [1, T]
            labels: [1]
        """
        # get audio paths and labels from the dataset
        example = self.dataset[index]
        audio_path = example["audio_path"]  
        audio_file = os.path.join(self.base_audio_dir, audio_path)
        
        audio = self.load_audio(audio_file)
        label = torch.tensor([example['target']], dtype=torch.int64)


        return {"audio": audio, "labels": label, "audio_length": audio.shape[1]}

    def load_audio(
        self, 
        audio_file
    ):
        """
        input:
            audio_file:one of audio_file path
        return:
            audio:[T]
              T:audio timestep
        """
        audio, _ = torchaudio.load(audio_file)
        if audio.shape[0] > 1 and self.is_mono:
            audio = torch.mean(audio, dim=0, keepdim=True)

        return audio

    def collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size , target_length)
            labels_tensor:(batch_size)
        """
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]
        
        audio_tensor = pad_sequence(audio_list, batch_first=True)
        audio_length_tensor = torch.tensor(audio_length_list)
        label_tensor = torch.cat(label_list, dim=0)

        if audio_tensor.shape[-1] < self.target_sec * self.sample_rate:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.target_sec * self.sample_rate - audio_tensor.shape[-1], 0, 0))
        elif audio_tensor.shape[-1] > self.target_sec * self.sample_rate:
            audio_tensor = audio_tensor[:, :self.target_sec * self.sample_rate]
        
        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "audio_length": audio_length_tensor,
        }

class ESC50dataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_args, 
        codec_name,
        train_split: 0.9,
        test_split: 0.1,
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=16,
        train_num_workers=4,
        val_num_workers=4,
        test_num_workers=4
    ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_split = train_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.dataset = ESC50dataset(**self.dataset_args)
        self.train_size = int(len(self.dataset) * self.train_split)
        self.test_size = len(self.dataset) - self.train_size
        self.train_dataset, self.test_dataset = random_split(self.dataset, [self.train_size, self.test_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.val_num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.test_num_workers,
        )