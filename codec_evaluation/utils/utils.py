import os
import numpy as np
import torch.nn.functional as F  
from codec_evaluation.codecs.encodec import Encodec
from codec_evaluation.codecs.mimi import Mimi
from codec_evaluation.codecs.semanticodec import SemantiCodec
from codec_evaluation.codecs.speechtokenizer import SpeechTokenizer
from codec_evaluation.codecs.wavlm_kmeans import WavLMKmeans
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
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if os.path.splitext(file)[1] in exts:
                    audio_files.append(os.path.join(root, file))
        return audio_files

def init_codec(modelname, sample_rate, mode, device, freeze = False):
    """
        Codec initialization
        input:
            modelname: codecname
            sample_rate: The sample rate of the input audio
            mode: "quantized_emb" "unquantized_emb",etc.
            device: Select the device to use
            freeze: Whether to calculate the gradient
        return:
            model: Initialied codec

    """
    if modelname == 'dac':
        model = DAC(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)  
    elif modelname == 'encodec':
        model = Encodec(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)
    elif modelname == 'mimi':
        model = Mimi(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)
    elif modelname == 'semanticodec':
        model = SemantiCodec(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)
    elif modelname =='speechtokenizer':
        model = SpeechTokenizer(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)
    elif modelname =='wavlm_kmeans':
        model = WavLMKmeans(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)
    elif modelname =='wavtokenizer':
        model = WavTokenizer(
            sample_rate=sample_rate, 
            mode=mode
        ).to(device)
    else:
        raise ValueError(f"Invalid model name: {modelname}")

    if freeze: 
        for name, params in model.named_parameters():
            params.requires_grad = False

    return model

def cut_or_pad(waveform, target_length, codecname = None):
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
        
        # Cut or pad
        if waveform_length > target_length:
            if waveform.dim() == 2:
                # Random cut for 2D input
                start = np.random.randint(0, waveform_length - target_length)
                waveform = waveform[:, start:start + target_length]
            else:
                # Direct cut for higher dimensions
                waveform = waveform[..., :target_length]
        elif waveform_length < target_length:
            # Pad
            padding_length = target_length - waveform_length
            waveform = F.pad(waveform, (0, padding_length))
        
        return waveform


def find_lastest_ckpt(directory):
    if directory is None:
        return None
    ckpt_file = glob.glob(os.path.join(directory, "*.ckpt"))

    if not ckpt_file:
        log.info(f"No ckpt files found in this directory: {directory}")
        return None

    latest_ckpt_file = max(ckpt_file, key=os.path.getmtime)
    return latest_ckpt_file

        