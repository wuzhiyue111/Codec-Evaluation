from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset 
from einops import rearrange, repeat
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class GSdataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,
        is_mono,
        dataset_path,  
        base_audio_dir=None,  
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.dataset_path = dataset_path
        self.base_audio_dir = base_audio_dir   

        self.dataset = load_from_disk(dataset_path).filter(lambda x: x["split"] == split)
        
        # tag mapping
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
            audio: [1,T]
            label: [1]
        """
        example = self.dataset[index]
        audio_path = example["wav"]   

        if self.base_audio_dir:
            audio_path = str(Path(self.base_audio_dir) / Path(audio_path))
        
        audio = self.load_audio(audio_path)
        label = torch.tensor([self.class2id[example["y"]]]) 
        
        return {"audio": audio, "labels": label, "audio_length": audio.shape[1]}

    def load_audio(
        self, 
        audio_file
    ):
        """
        input:
            audio_file:one of audio_file path
        return:
            audio:[T]
              T:audio timestep
        """
        audio, _ = torchaudio.load(audio_file)
        if audio.shape[0] > 1 and self.is_mono:
            audio = torch.mean(audio, dim=0, keepdim=True)

        return audio

    def collate_fn(self, batch):
    
        """
        return:
            features_tensor:(batch_size * split_count, target_length)
            labels_tensor:(batch_size * split_count)
                split_count: the split count of audio
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