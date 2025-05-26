import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from datasets import load_from_disk
from codec_evaluation.utils.utils import cut_or_pad
from codec_evaluation.utils.logger import RankedLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = RankedLogger(__name__, rank_zero_only=True)

class EMOdataset(Dataset):
    def __init__(
        self,
        split,              
        dataset_path,       
        base_audio_dir,     
        sample_rate,
        target_sec,
        is_mono,
        task="regression",
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * sample_rate
        self.is_mono = is_mono
        self.task = task
        self.base_audio_dir = base_audio_dir  # 绝对路径，如音频文件根目录
        self.dataset = load_from_disk(dataset_path)
        self.dataset = self.dataset.filter(lambda x: x["split"] == split)
        self.classes = ["arousal", "valence"]  # 保持与原代码一致
        self.class2id = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            example = self.dataset[index]
            audio_path = example["wav"]  # 数据集中的相对音频路径
            audio_file = os.path.join(self.base_audio_dir, audio_path)
            logger.error(f"Error loading {audio_file}: {e}")
            return None

    def get_item(self, index):
        example = self.dataset[index]
        audio_path = example["wav"]  
        audio_file = os.path.join(self.base_audio_dir, audio_path)  # 拼接绝对路径
        
        segments, pad_mask = self.load_audio(audio_file)
        labels = torch.from_numpy(np.array(example["y"], dtype=np.float32))  # 直接从数据集中获取标签
        
        segments = torch.vstack(segments)
        return {
            "audio": segments,
            "labels": labels,
            "n_segments": len(pad_mask)
        }

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
        
        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length, task=self.task)
        
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
        dataset_args,  # 包含dataset_path、base_audio_dir等参数
        codec_name,
        train_batch_size=16,
        valid_batch_size=16,
        test_batch_size=16,
        train_num_workers=4,
        valid_num_workers=4,
        test_num_workers=4,
    ):
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