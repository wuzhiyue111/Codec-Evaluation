import torch
import json
import os
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from codec_evaluation.utils.utils import find_audios

logger = RankedLogger(__name__, rank_zero_only=True)


class Muchin_ctc_dataset(Dataset):
    def __init__(
        self,
        audio_dir,
        meta_path
    ):
        self.audio_dir = audio_dir

        with open(meta_path) as f:
            self.metadata = json.load(f)
        
        logger.info(f"Found {len(self.metadata)} metadata in {meta_path}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print(f"Error loading audio file {self.metadata[index]['filename']}: {e}")
            return None

    def get_item(self, index):
        """
        return:
            segments: [1, wavform_length]
            labels: [1, 20]
        """
        audio_file = self.metadata[index]['filename']
        audio_path = os.path.join(self.audio_dir, audio_file)
        waveform, _ = torchaudio.load(audio_path)  # [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        text = self.metadata[index]['lyric']
        return {"audio": waveform, "text": text, "audio_length": waveform.shape[1]}

    def collate_fn(self, batch):
        """
        return:
            audio: [B, T]
            text: [B]
        """
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        text_list = [item["text"] for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]

        audio_tensor = pad_sequence(audio_list, batch_first=True) # [B, T]
        audio_length_tensor = torch.tensor(audio_length_list)

        return {
            "audio": audio_tensor,
            "text": text_list,
            "audio_length": audio_length_tensor,
        }


class Muchin_ctc_module(pl.LightningDataModule):
    def __init__(
        self,
        audio_dir,
        meta_path,
        train_split: 0.9,
        test_split: 0.1,
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
        self.train_split = train_split
        self.test_split = test_split
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers
        self.dataset = Muchin_ctc_dataset(audio_dir, meta_path)
        self.train_size = int(len(self.dataset) * self.train_split)
        self.test_size = len(self.dataset) - self.train_size

    def setup(self, stage=None):
        train_dataset, test_dataset = random_split(self.dataset, 
                                                                  [self.train_size, self.test_size])
        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.valid_dataset = test_dataset
        if stage == "val":
            self.valid_dataset = test_dataset
        if stage == "test":
            self.test_dataset = test_dataset

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
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.valid_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
            num_workers=self.test_num_workers,
        )