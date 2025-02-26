import os
import torch
import torchaudio
import numpy as np
import json
import sys
from torch.utils.data import Dataset
sys.path.append(os.path.expanduser('~/project/Codec-Evaluation'))
from codec_evaluation.evaluation.utils import find_audios, cut_or_pad


class EMOdataset(Dataset):
    """
        
    """
    def __init__(self, sample_rate, target_sec, n_segments, is_mono, is_normalize, audio_dir, meta_dir, device):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.n_segments = n_segments
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.is_normalize = is_normalize
        self.audio_files = find_audios(audio_dir)
        self.meta_dir = meta_dir
        self.device = device

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        
        return self.get_item(index)
        
    def get_item(self, index):
        
        """
            return：
                segments：[n_segments, segments_length]
                labels：[n_segments,2]
        """
        audio_file = self.audio_files[index]
        waveform = self.load_audio(audio_file)
        label = self.load_label(audio_file)

        segments = self.split_audio(waveform, self.n_segments)
        segments = torch.stack(segments, dim=0) 

        labels = label.unsqueeze(0).repeat(self.n_segments, 1)   

        return segments, labels

    def load_audio(
        self,
        audio_file,
    ):
        """
            input:
                audio_file:one of audio_file path
            return:
                waveform:[T]
        """
        if len(audio_file) == 0:
            raise FileNotFoundError("No audio files found in the specified directory.")
       
        try:
            waveform, _ = torchaudio.load(audio_file)
        except Exception as e:
            print(f"Error loading audio file {audio_file}: {e}")
            return None
            
        # Convert to mono if needed
        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Normalize to [-1, 1] if needed
        if self.is_normalize:
            waveform = waveform / waveform.abs().max()

        waveform = cut_or_pad(waveform = waveform, 
                              target_length = self.target_length)
        
        waveform = waveform.squeeze(0)  
        waveform = waveform.to(self.device)
        
        return waveform

    def load_label(self, audio_file):

        """Load the label for a given audio from meta.json.
            input:
                audio_file:one of audio_file path
            return:
                label:[2]
        """
        metadata_path = os.path.join(self.meta_dir, 'meta.json')
        with open(metadata_path) as f:
            metadata = json.load(f)

        audio_name_without_ext = os.path.splitext(os.path.basename(audio_file))[0]      #Extract the filename without the extension
        label = torch.from_numpy(np.array(metadata[audio_name_without_ext]['y'], dtype=np.float32))   
        label = label.to(self.device)

        return label
    
    def split_audio(self, waveform, n_segments):
        """
            input:
                waveform:[T];
                n_segments: the number of segments per audio
            return:
                segments:[n_segments,segment_length]
        """
        
        segment_length = self.target_length // n_segments  # length of per segment
        segments = []

        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = waveform[start:end]  
            segments.append(segment)

        return segments