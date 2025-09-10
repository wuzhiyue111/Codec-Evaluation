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
import matplotlib.pyplot as plt
import codec_evaluation
root_path = codec_evaluation.__path__[0]

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


def plot_mrc_avg_same_id(avg_same_id_dict, codec_name, task):
        """Plot the average codebook same ID across 10 rounds for each codebook.

        inputs:
            avg_same_id_dict (dict): A dictionary containing the average codebook same ID for each codebook across 10 rounds.
            codec_name (str): The name of the codec.
            task (str): The task name.
        returns:
            photo
        """
        num_codebooks = len(avg_same_id_dict)
        num_rounds = len(next(iter(avg_same_id_dict.values())))
        bar_width = 0.8 / num_rounds
        x_labels = list(avg_same_id_dict.keys())
        x_positions = np.arange(num_codebooks)
        plt.figure(figsize=(12, 6))
        for i in range(num_rounds):
            values = [float(avg_same_id_dict[codebook][i].strip('%')) for codebook in x_labels]
            positions = x_positions + i * bar_width
            plt.bar(positions, values, width=bar_width, label=f"Round {i + 1}")

        plt.xticks(x_positions + (bar_width * (num_rounds - 1)) / 2, x_labels)
        plt.xlabel("Codebooks")
        plt.ylabel("Codebook ID Proportion(%)", rotation=90, labelpad=3, fontweight='bold')
        plt.title(f"{codec_name} - {task} - Average Codebook Same ID Across 10 Rounds")
        plt.legend(title="Reconstruction Rounds")
        plt.ylim(0, 100)
        y_ticks = np.arange(0, 101, 10)
        plt.yticks(y_ticks, [f"{tick}%" for tick in y_ticks])
        save_dir = os.path.join(root_path, "id_sensitive", f"{task}_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{codec_name}_mrc_avg_same_id.png")
        plt.savefig(save_path)

def plot_os_avg_same_id(percent_same_id_avg_list, num_codebooks, codec_name, task):
        """Plot the average codebook same ID for each codebook offsetting by *ms.

        inputs:
            percent_same_id_avg_list (list): A list containing the average codebook same ID for each codebook offsetting by *ms.
            num_codebooks (int): The number of codebooks.
            codec_name (str): The name of the codec.
            task (str): The task name.
        returns:
            photo
        """
        x_labels = ["Codebook Same ID"]
        codebook_labels = [f"codebook{i + 1}" for i in range(num_codebooks)]
        bar_width = 0.1  
        percent_same_id_avg_list = [float(p.strip('%')) / 100 for p in percent_same_id_avg_list]
        _, ax = plt.subplots()

        total_display_width = 0.8
        total_width = bar_width * num_codebooks
        offset = (total_display_width - total_width) / 2  

        for i in range(num_codebooks):
            positions = [offset + j + i * bar_width for j in range(len(x_labels))]
            values = [percent_same_id_avg_list[i]]
            bars = ax.bar(positions, values, width=bar_width, label=codebook_labels[i])
            ax.bar_label(bars, padding=3, labels=[f"{percent_same_id_avg_list[i] * 100:.1f}"])

        ax.set_xticks([offset + i + (bar_width * (num_codebooks - 1)) / 2 for i in range(len(x_labels))])
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Codebook ID Proportion(%)", rotation=90, labelpad=3, fontweight='bold')
        ax.set_title(f"{codec_name} - {task} - Codebook Same Id Average(%)")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.set_yticklabels([f"{i * 20}" for i in range(6)])
        save_dir = os.path.join(root_path, "id_sensitive", f"{task}_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{codec_name}_os_avg_same_id.png")
        plt.savefig(save_path)


    