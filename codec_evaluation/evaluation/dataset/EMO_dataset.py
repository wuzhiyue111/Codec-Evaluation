import os, random
import torch
import torchaudio
import numpy as np
import argparse
import json
from mido import MidiFile
np.set_printoptions(precision=4, suppress=True)

def load_audio(
    is_mono=True, 
    is_normalize=False, 
    device=torch.device('cpu'),
    parent_dir='/home/wsy/project/MARBLE-Benchmark/data/EMO/emomusic/wav'  # 设置默认值
):
    """Load audio file from a directory and convert to target sample rate.
    Converts audio to mono if is_mono is True, and normalizes if is_normalize is True.
    This function now calls `find_audios` to get the files instead of receiving a file path directly.

    Args:
        target_sr (int): target sample rate, if not equal to sample rate of audio file, resample to target_sr
        is_mono (bool, optional): convert to mono. Defaults to True.
        is_normalize (bool, optional): normalize to [-1, 1]. Defaults to False.
        device (torch.device, optional): device to use for resampling. Defaults to torch.device('cpu').
        parent_dir (str, optional): the directory to search for audio files. Defaults to './audio_files'.
    
    Returns:
        torch.Tensor: waveform of shape (1, n_sample) of the first found audio file.
    """
    # Call find_audios to find audio files, now looking only for .wav files
    audio_files = find_audios(parent_dir, exts=['.wav'])
    
    if len(audio_files) == 0:
        raise FileNotFoundError("No audio files found in the specified directory.")

    # Use the first audio file found
    file_path = audio_files[0]
    print(f"Loading audio file: {file_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1 and is_mono:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalize to [-1, 1] if needed
    if is_normalize:
        waveform = waveform / waveform.abs().max()
    
    # Move the waveform to the desired device (e.g., CPU or GPU)
    waveform = waveform.to(device)
    
    return waveform


def load_label(audio_name_without_ext):
    """Load the label for a given audio from meta.json."""
    
    metadata_dir = '/home/wsy/project/MARBLE-Benchmark/data/EMO/emomusic'
    metadata_path = os.path.join(metadata_dir, 'meta.json')
    with open(metadata_path) as f:
        metadata = json.load(f)

    label = torch.from_numpy(np.array(metadata[audio_name_without_ext]['y'], dtype=np.float32))
    return label

def find_audios(
    parent_dir, 
    exts=['.wav', '.mp3', '.flac', '.webm', '.mp4']
):
    """Find all audio files with the specified extensions in the given directory.

    Args:
        parent_dir (str): Directory to search for audio files.
        exts (list, optional): List of valid audio file extensions to search for. Defaults to audio file types like '.wav', '.mp3', etc.
    
    Returns:
        list: List of audio file paths that match the extensions.
    """
    audio_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if os.path.splitext(file)[1] in exts:
                audio_files.append(os.path.join(root, file))
    return audio_files