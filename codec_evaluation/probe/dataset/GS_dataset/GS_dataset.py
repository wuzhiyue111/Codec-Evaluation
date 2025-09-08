from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_from_disk
from codec_evaluation.utils.utils import cut_or_pad
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class GSdataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,
        is_mono,
        dataset_path,  # 替换原 audio_dir 和 meta_path，指向 .arrow 数据集路径
        base_audio_dir=None,  # 可选：如果路径是相对的，指定音频文件的基础目录
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.dataset_path = dataset_path
        self.base_audio_dir = base_audio_dir  # 用于拼接相对路径
        
        # 加载 .arrow 数据集
        self.dataset = load_from_disk(dataset_path).filter(lambda x: x["split"] == split)
        
        # 构建标签映射（假设数据集中已包含 class 信息，或需要从数据集中提取）
        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()} 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        """
        return:
            segments: [n_segments, segments_length]
            labels: [n_segments, 2]
        """
        example = self.dataset[index]
        audio_path = example["wav"]  # 假设数据集中音频路径字段为 "wav"
        
        # 拼接完整路径（如果是相对路径）
        if self.base_audio_dir:
            audio_path = str(Path(self.base_audio_dir) / Path(audio_path))
        
        segments, pad_mask = self.load_audio(audio_path)
        label = self.class2id[example["y"]]  
        labels = []
        for mask in pad_mask:
            if mask == 1:
                labels.append(label)
            else:
                labels.append(-100)
        labels = torch.tensor(labels)
        segments = torch.vstack(segments)
        
        return {"audio": segments, "labels": labels, "n_segments": len(pad_mask)}

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
        
        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length)
        return waveform, pad_mask

    def collate_fn(self, batch):
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]
        
        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.cat(label_list, dim=0)
        
        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }
    
class GSdataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_args, 
        codec_name,
        train_batch_size=32,
        val_batch_size=2,
        test_batch_size=16,
        train_num_workers=8,
        val_num_workers=4,
        test_num_workers=4,
    ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 加载训练/验证集对应的 .arrow 数据集
            self.train_dataset = GSdataset(split="train", **self.dataset_args)
            self.val_dataset = GSdataset(split="valid", **self.dataset_args)
        if stage == "val":
            self.val_dataset = GSdataset(split="valid", **self.dataset_args)
        if stage == "test":
            self.test_dataset = GSdataset(split="test", **self.dataset_args)

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
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.val_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )