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

class MTGGenredataset(Dataset):
    def __init__(
        self,
        base_audio_dir,  # 音频文件绝对根目录
        dataset_path,       # .arrow数据集目录
        sample_rate,
        target_sec,
        is_mono=True,      
    ):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)  # 加载.arrow数据集
        self.base_audio_dir = base_audio_dir
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = sample_rate * target_sec  # 目标片段的采样点数
        self.is_mono = is_mono
        print(f"Found {len(self.dataset)} audio files in {base_audio_dir}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            logger.error(f"Error loading audio file {self.dataset[index]['audio_path']}: {e}")
            return None  

    def get_item(self, index):
        sample = self.dataset[index]
        audio_path = os.path.join(self.base_audio_dir, sample["audio_path"])
        
        # 加载音频
        waveform, _ = torchaudio.load(audio_path)
        if self.is_mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # 转单声道
        
        # 获取原始音频长度（此时可确定orig_length >= 30秒）
        orig_length = waveform.shape[1]
        
        # 随机截取15秒片段
        max_start = orig_length - self.target_length
        start = random.randint(0, max_start)
        end = start + self.target_length
        waveform = waveform[:, start:end]  # 随机位置截取
        audio_length = self.target_length  # 截取后的长度
        
        return {
            "audio": waveform,           
            "audio_length": audio_length,  # 始终为target_length
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
    
class MTGGenreModule(pl.LightningDataModule):
    def __init__(
        self,
        train_audio_dir=None,
        valid_audio_dir=None,
        test_audio_dir=None,
        base_audio_dir=None,
        sample_rate=44100,
        target_sec=15,  # 目标片段长度（秒）
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
        self.train_audio_dir = train_audio_dir
        self.valid_audio_dir = valid_audio_dir
        self.test_audio_dir = test_audio_dir
        self.base_audio_dir = base_audio_dir
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.is_mono = is_mono

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MTGGenredataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.train_audio_dir,
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)
            self.valid_dataset = MTGGenredataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.valid_audio_dir,
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)
        if stage == "val":
            self.valid_dataset = MTGGenredataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.valid_audio_dir,
                sample_rate=self.sample_rate,
                target_sec=self.target_sec,
                is_mono=self.is_mono)
        if stage == "test":
            self.test_dataset = MTGGenredataset(
                base_audio_dir=self.base_audio_dir,
                dataset_path=self.test_audio_dir,
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