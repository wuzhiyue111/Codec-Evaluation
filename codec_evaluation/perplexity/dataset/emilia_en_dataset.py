import torch
import os
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk

logger = RankedLogger(__name__, rank_zero_only=True)

class Emilia_ppl_dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,       # .arrow文件路径
        base_audio_dir: str        # 音频文件根目录（用于拼接路径）
    ):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)  # 加载.arrow数据集
        self.base_audio_dir = base_audio_dir                # 音频根目录
        self.dataset_path = dataset_path
        print(f"Found {len(self.dataset)} audio files in {base_audio_dir}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            audio_file = self.dataset[index]['wav']
            audio_path = os.path.join(self.base_audio_dir, audio_file)
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return None
        
    def get_item(self, index):
        sample = self.dataset[index]
        audio_path = os.path.join(self.base_audio_dir, sample["wav"])  # 拼接完整路径
            
        # 加载音频
        waveform, _ = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # 转单声道
            
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

class Emilia_ppl_Module(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        base_audio_dir=None,
        train_ratio=0.9,
        test_ratio=0.1,
        train_batch_size=16,
        valid_batch_size=16,
        test_batch_size=16,
        train_num_workers=4,
        valid_num_workers=4,
        test_num_workers=4,
        seed=42,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.base_audio_dir = base_audio_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers
        self.seed = seed
        self.full_dataset = Emilia_ppl_dataset(self.dataset_path, self.base_audio_dir)
        self.train_size = int(self.train_ratio * len(self.full_dataset))
        self.test_size = len(self.full_dataset) - self.train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.full_dataset,[self.train_size, self.test_size],)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.full_dataset.collate_fn,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            collate_fn=self.full_dataset.collate_fn,
            num_workers=self.valid_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.full_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )