import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from codec_evaluation.utils.logger import RankedLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = RankedLogger(__name__, rank_zero_only=True)

class GTZANdataset(Dataset):
    def __init__(
        self,
        dataset_path,  # .arrow数据集路径（如"/path/to/GTZAN_train_dataset"）
    ):
        self.dataset = load_from_disk(dataset_path)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            audio_path = self.dataset[index]["audio_path"]  # 数据集中的相对路径
            logger.error(f"Error loading {audio_path}: {e}")
            return None  

    def get_item(self, index):
        """
            return:
                segments: [n_segments, segments_length]
        """
        example = self.dataset[index]
        waveform_np = torch.from_numpy(example["audio"]["array"])
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=0)
        waveform = waveform_np.float().unsqueeze(0)  # 转换为 [1, T] 形状
        
        return {"audio": waveform}

    def collate_fn(self, batch):
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        audio_tensor = pad_sequence(audio_list, batch_first=True)

        return {"audio": audio_tensor}

class GTZANdataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_args,
        train_audio_dir,
        valid_audio_dir,
        test_audio_dir,
        codec_name,
        train_batch_size=32,
        valid_batch_size=2,
        test_batch_size=16,
        train_num_workers=8,
        valid_num_workers=4,
        test_num_workers=4,
    ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_audio_dir = train_audio_dir
        self.valid_audio_dir = valid_audio_dir
        self.test_audio_dir = test_audio_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GTZANdataset(dataset_path=self.test_audio_dir, **self.dataset_args)  
            self.valid_dataset = GTZANdataset(dataset_path=self.valid_audio_dir, **self.dataset_args) 
        if stage == "val":
            self.valid_dataset = GTZANdataset(dataset_path=self.valid_audio_dir, **self.dataset_args) 
        if stage == "test":
            self.test_dataset = GTZANdataset(dataset_path=self.test_audio_dir, **self.dataset_args)

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