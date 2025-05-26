import os
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_from_disk
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.utils import cut_or_pad

import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = RankedLogger(__name__, rank_zero_only=True)

class MTGTop50dataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,
        is_mono,
        base_audio_dir,  # 音频文件绝对根目录
        dataset_path,       # .arrow数据集目录
        task,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = target_sec * sample_rate
        self.is_mono = is_mono
        self.base_audio_dir = base_audio_dir
        self.dataset_path = dataset_path
        self.task = task

        # 加载.arrow数据集
        dataset_path = os.path.join(dataset_path, f"MTGTop50_{split}_dataset")
        self.dataset = load_from_disk(dataset_path)
        
        self.class2id = self.read_class2id()
        self.id2class = {v: k for k, v in self.class2id.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            audio_path = self.dataset[index]["audio_path"]  # 数据集中的相对路径
            full_path = os.path.join(self.base_audio_dir, audio_path)
            logger.error(f"Error loading {full_path}: {e}")
            return None  

    def get_item(self, index):
        """
        return:
            segments: [1, wavform_length]
            labels: [1, 50]
        """
        record = self.dataset[index]
        relative_audio_path = record["audio_path"]
        tags = record["TAGS"].split()  # 从.arrow读取标签（空格分隔）
        
        # 拼接绝对路径
        audio_file = os.path.join(self.base_audio_dir, relative_audio_path)
        
        # 加载和处理音频
        segments, pad_mask = self.load_audio(audio_file)
        
        # 构建标签
        label = torch.zeros(len(self.class2id))
        for tag in tags:
            if tag.strip() in self.class2id:
                label[self.class2id[tag.strip()]] = 1

        segments = torch.vstack(segments)

        return {"audio": segments, "labels": label, "n_segments": len(pad_mask)}
    
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
        
        if waveform.shape[1] > 150 * self.sample_rate:
            waveform = waveform[:, :150 * self.sample_rate]

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length, task=self.task)

        return waveform, pad_mask 

    def read_class2id(self):
        """从.arrow数据集中构建class2id映射"""
        class2id = {}
        
        # 只从训练集和验证集构建完整标签映射
        for split in ['train', 'validation']:
            dataset = load_from_disk(os.path.join(self.dataset_path, f"MTGTop50_{split}_dataset"))
            for record in dataset:
                tags = record["TAGS"].split()  # 从.arrow读取标签
                for tag in tags:
                    tag = tag.strip()
                    if tag and tag not in class2id:     # 过滤空字符串，并检查唯一性
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
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )