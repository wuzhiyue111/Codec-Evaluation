import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from datasets import load_from_disk
from codec_evaluation.utils.utils import cut_or_pad
from codec_evaluation.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

class ESC50dataset(Dataset):
    def __init__(
        self,
        sample_rate,
        target_sec,
        is_mono,
        dataset_path,  # 修改：接收保存的数据集路径
        task,
        base_audio_dir,  # 可选：如果使用相对路径，指定音频文件的基础目录
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.task = task
        self.base_audio_dir = base_audio_dir
        
        # 加载保存的数据集
        self.dataset = load_from_disk(dataset_path)
        # 假设数据集中包含 'wav' 列（音频路径）和 'id' 列（标签）
    
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
            segments: [n_segments, segments_length]
            labels: [n_segments, 2]
        """
        # 从数据集中获取音频路径和标签
        example = self.dataset[index]
        audio_path = example["audio_path"]  # 数据集中的相对路径（如"blues/blues.00037.wav"）
        audio_file = os.path.join(self.base_audio_dir, audio_path)
        
        segments, pad_mask = self.load_audio(audio_file)
        label = torch.tensor([example['target']], dtype=torch.int64)

        segments = torch.vstack(segments)

        return {"audio": segments, "labels": label, "n_segments": len(pad_mask)}

    def load_audio(
        self, 
        audio_file
    ):
        """
        input:
            audio_file: one of audio_file path
        return:
            waveform: [T]
        """
        waveform, _ = torchaudio.load(audio_file)

        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length, task=self.task)

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

class ESC50dataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_args, 
        codec_name,
        train_split: 0.9,
        test_split: 0.1,
        train_batch_size=16,
        valid_batch_size=16,
        test_batch_size=16,
        train_num_workers=4,
        valid_num_workers=4,
        test_num_workers=4
    ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_split = train_split
        self.test_split = test_split
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
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