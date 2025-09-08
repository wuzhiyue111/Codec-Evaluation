import torch
import os
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk

logger = RankedLogger(__name__, rank_zero_only=True)

class Common_voice_dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,       # .arrow文件路径
        base_audio_dir: str,        # 音频文件根目录（用于拼接路径）
        target_samplerate=48000
        
    ):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)  # 加载.arrow数据集
        self.base_audio_dir = base_audio_dir                # 音频根目录
        self.dataset_path = dataset_path
        self.target_samplerate = target_samplerate
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
        audio_path = os.path.join(self.base_audio_dir, sample["audio_path"])  # 拼接完整路径
            
        # 加载音频
        waveform, samplerate = torchaudio.load(audio_path)

        if samplerate != self.target_samplerate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=samplerate, 
                new_freq=self.target_samplerate
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # 转单声道

        if waveform.shape[1] > 20*48000:
            return None
        else:
            return {
                "audio": waveform,
                "text": sample["text"],
                "audio_length": waveform.shape[1]
            }

    def collate_fn(self, batch):
        """
        return:
            audio: [B, T]
            text: [B]
        """
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        text_list = [item["text"] for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]
        
        audio_tensor = pad_sequence(audio_list, batch_first=True)
        audio_length_tensor = torch.tensor(audio_length_list)
        
        return {
            "audio": audio_tensor,
            "text": text_list,
            "audio_length": audio_length_tensor,
        }

class Common_voice_module(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        base_audio_dir,
        target_samplerate,
        train_split: 0.98,
        val_split:0.01,
        test_split: 0.01,
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=1,
        train_num_workers=1,
        val_num_workers=1,
        test_num_workers=1,
        
    ):
        super().__init__()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.target_samplerate = target_samplerate
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.dataset = Common_voice_dataset(dataset_path, base_audio_dir, self.target_samplerate)
        self.train_size = int(len(self.dataset) * self.train_split)
        self.val_size = int(len(self.dataset) * self.val_split)
        self.test_size = len(self.dataset) - self.val_size - self.train_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, 
                                                                  [self.train_size, self.val_size, self.test_size])

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
            dataset=self.val_dataset,
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