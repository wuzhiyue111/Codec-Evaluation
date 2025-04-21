import os
import torch
import torchaudio
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger

from codec_evaluation.utils.utils import cut_or_pad
logger = RankedLogger(__name__, rank_zero_only=True)

class EMOdataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,
        is_mono,
        audio_dir,
        meta_dir,
        task = "regression",
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.audio_dir = audio_dir
        self.task = task

        self.meta_dir = meta_dir
        with open(os.path.join(self.meta_dir, "meta.json")) as f:
            self.metadata = json.load(f)
        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.classes = """arousal, valence""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.audio_names_without_ext)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            logger.error(f"Error loading audio file {self.audio_names_without_ext[index]}.wav:{e}")
            return None

    def get_item(self, index):
        """
        return:
            segments: [n_segments, segments_length]
            labels: [n_segments, 2]
        """
        audio_name_without_ext = self.audio_names_without_ext[index]
        audio_path = audio_name_without_ext + '.wav'
        audio_file = os.path.join(self.audio_dir, audio_path)

        segments, pad_mask= self.load_audio(audio_file)
        labels = torch.from_numpy(np.array(self.metadata[audio_name_without_ext]['y'], dtype=np.float32))

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
            waveform:[n,T]
        """

        waveform, _ = torchaudio.load(audio_file)

        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length, task = self.task)

        return waveform, pad_mask 

    
    def collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size * n_segments, length)
            labels_tensor:(batch_size * n_segments, 2)
            if codecname='semanticodec'
                labels_tensor:(batch_size * n_segments * 2, 2)
        """
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]
        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.vstack(label_list)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }

class EMOdataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_args, 
            codec_name,
            train_batch_size=16,
            valid_batch_size=16,
            test_batch_size=16,
            train_num_workers=4,
            valid_num_workers=4,
            test_num_workers=4):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = EMOdataset(split="train", **self.dataset_args)
            self.valid_dataset = EMOdataset(split="valid", **self.dataset_args)
        if stage == "val":
            self.valid_dataset = EMOdataset(split="valid", **self.dataset_args)
        if stage == "test":
            self.test_dataset = EMOdataset(split="test", **self.dataset_args)

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