import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from einops import rearrange, repeat
from torch.nn.utils.rnn import pad_sequence
from codec_evaluation.utils.logger import RankedLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = RankedLogger(__name__, rank_zero_only=True)

class GTZANdataset(Dataset):
    def __init__(
        self,
        dataset_path,
        sample_rate,
        target_sec,
    ):
        """
        split: train/valid/test
        dataset_path: .arrow dataset path
        base_audio_dir: root path for audio files
        """
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        
        self.dataset = load_from_disk(dataset_path)
        
        self.class2id = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 
                         'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            example = self.dataset[index]
            audio_path = example["audio_path"]
            logger.error(f"Error loading {audio_path}: {e}")
            return None

    def get_item(self, index):
        """
            return:
                audio: [1,T]
                labels: [1]
        """
        example = self.dataset[index]
        audio = torch.from_numpy(example["audio"]["array"])
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = audio.float().unsqueeze(0)
        label = torch.tensor([self.class2id[example["labels"]]]) 
        
        return {"audio": audio, "labels": label, "audio_length": audio.shape[1]}

    def collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size * split_count, target_length)
            labels_tensor:(batch_size * split_count)
        """
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]
        
        audio_tensor = pad_sequence(audio_list, batch_first=True)
        audio_length_tensor = torch.tensor(audio_length_list)
        label_tensor = torch.cat(label_list, dim=0)

        split_count = round(audio_tensor.shape[-1] / self.target_length)

        if audio_tensor.shape[-1] < self.target_length * split_count:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.target_length * split_count - audio_tensor.shape[-1], 0, 0))
        elif audio_tensor.shape[-1] > self.target_length * split_count:
            audio_tensor = audio_tensor[:, :self.target_length * split_count]

        audio_tensor = rearrange(audio_tensor, 'b (n t) -> (b n) t', n=split_count, t=self.target_length)
        label_tensor = repeat(label_tensor, 'b -> (b n)', n=split_count)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "audio_length": audio_length_tensor,
        }

class GTZANdataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_args,
        train_audio_dir,
        val_audio_dir,
        test_audio_dir,
        train_batch_size=32,
        val_batch_size=2,
        test_batch_size=16,
        train_num_workers=8,
        val_num_workers=4,
        test_num_workers=4,
    ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_audio_dir = train_audio_dir
        self.val_audio_dir = val_audio_dir
        self.test_audio_dir = test_audio_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GTZANdataset(dataset_path=self.train_audio_dir, **self.dataset_args)  
            self.val_dataset = GTZANdataset(dataset_path=self.val_audio_dir, **self.dataset_args) 
        if stage == "val":
            self.val_dataset = GTZANdataset(dataset_path=self.val_audio_dir, **self.dataset_args) 
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