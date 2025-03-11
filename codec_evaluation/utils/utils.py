import os
import numpy as np
import torch.nn.functional as F  
from codec_evaluation.codecs.encodec import Encodec
from codec_evaluation.codecs.mimi import Mimi
from codec_evaluation.codecs.semanticodec import SemantiCodec
from codec_evaluation.codecs.speechtokenizer import SpeechTokenizer
from codec_evaluation.codecs.wavtokenizer import WavTokenizer
from codec_evaluation.codecs.dac import DAC
import glob
import logging

log = logging.getLogger(__name__)

def find_audios(
        audio_dir, 
        exts=['.wav', '.mp3', '.flac', '.webm', '.mp4']
    ):
        """Find all audio files with the specified extensions in the given directory.

        inputs:
            audio_dir (str): Directory to search for audio files.
            exts (list, optional): List of valid audio file extensions to search for. Defaults to audio file types like '.wav', '.mp3', etc.
        
        returns:
            list: List of audio file paths that match the extensions.
        """
        audio_files = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if os.path.splitext(file)[1] in exts:
                    audio_files.append(os.path.join(root, file))
        return audio_files

def cut_or_pad(waveform, target_length, task=None, codecname = None):
        """Cut or pad a waveform or a feature to a target length."""
        
        if codecname == 'semanticodec':
            if waveform.dim() == 4:
                waveform = waveform.permute(0, 3, 2, 1)  # [B, T, 2, D] -> [B, D, 2, T]
            elif waveform.dim() == 3:
                waveform = waveform.permute(0, 2, 1)  # [B, T, D] -> [B, D, T]
    
    # Get waveform length based on dimension
        if waveform.dim() == 2:
            waveform_length = waveform.shape[1]  # [B, T]
        elif waveform.dim() == 3:
            waveform_length = waveform.shape[2]  # [B, D, T]
        elif waveform.dim() == 4:
            waveform_length = waveform.shape[3]  # [B, D, 2, T]
        else:
            raise ValueError("Unsupported waveform dimension!")
        
        if waveform.dim() == 2: 

            segments, pad_mask = split_audio(waveform = waveform, segment_length = target_length, task=task, pad_value=0)

            return segments, pad_mask
        else:
            if waveform_length > target_length:
                waveform = waveform[..., :target_length]
        
            return waveform

def find_lastest_ckpt(directory):
    if directory is None:
        return None
    search_path = os.path.join(directory, "*.ckpt")
    ckpt_file = glob.glob(search_path)

    if not ckpt_file:
        log.info(f"No ckpt files found in this directory: {search_path}")
        return None

    latest_ckpt_file = max(ckpt_file, key=os.path.getmtime)
    return latest_ckpt_file

def split_audio(waveform, segment_length, task, pad_value=-100):
    """
    input:
        waveform:[1,T];
        n_segments: the number of segments per audio
    return:
        segments:[n_segments,segment_length]
    """

    total_length = waveform.shape[1]  # 音频的总长度
    segments = []
    pad_mask = []

    if task != "regression":
        for start in range(0, total_length, segment_length):
            end = start + segment_length
            if end <= total_length:  
                segment = waveform[:, start:end] 
                pad_mask.append(1) 
            else:  
                segment = waveform[:, start:] 
                
                padding_length = segment_length - segment.shape[1]
                segment = F.pad(segment, (0, padding_length), value=pad_value)
                pad_mask.append(0) 

            segments.append(segment)
    else:
        for start in range(0, total_length, segment_length):
            end = start + segment_length
            if end <= total_length:  
                segment = waveform[:, start:end] 
                pad_mask.append(1) 
                segments.append(segment)

    return segments, pad_mask




    