import argparse
from codec_evaluation.init_codecs import init_codec
import torchaudio
import random
import numpy as np
import torch
import codec_evaluation
from typing import Optional
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
root_path = codec_evaluation.__path__[0]
from codec_evaluation.probe.dataset.LibriTTS_dataset.libritts_ctc import (
    LibriTTS_ctc_dataset,
)
from codec_evaluation.utils.utils import (
    plot_mrc_avg_same_id,
    plot_os_avg_same_id,
)

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

class IDSensitiveEvaluation:
    def __init__(
        self,
        codec_name: str,
        model_ckpt_dir: str,
        sample_rate: int,
        num_codebooks: int,
        need_resample: bool,
        dataset_audio_dir: str,
        device: str,
        task: str,
        batch_size: int = 24,
        num_workers: int = 8,
        shift_time: int = 2,
        subset_step: int = 1200,
        **kwargs,
    ):
        """
            codec_name: codec model name
            model_ckpt_dir: codec model checkpoint directory
            sample_rate: audio sample rate
            num_codebooks: number of codebooks
            need_resample: boolean, whether to resample the audio after decoding
            dataset_audio_dir: dataset audio directory
            task: task name "MRC" or "OS"  MRC: multi-round reconstruction, OS: offset
            num_workers: number of workers
            shift_time: shift time in ms
            kwargs: other arguments
        """
        assert task in ["MRC", "OS"], f"Invaild task: {task}. Task must be either 'MRC' or 'OS'." 

        self.device = device
        self.codec_encode = init_codec(
            modelname=codec_name,
            mode="encode",
            sample_rate=sample_rate,
            device=self.device,
            num_codebooks=num_codebooks,
            need_resample=need_resample,
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        )

        self.codec_reconstruct = init_codec(
            modelname=codec_name,
            mode="reconstruct",
            sample_rate=sample_rate,
            device=self.device,
            num_codebooks=num_codebooks,
            need_resample=need_resample,
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        )

        self.num_codebooks = num_codebooks
        self.codec_name = codec_name
        self.codec_sample_rate = self.codec_encode.orig_sample_rate
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = device
        self.shift_time = shift_time
        self.subset_step = subset_step
        self.task = task
        dataset = LibriTTS_ctc_dataset(audio_dir=dataset_audio_dir)
        dataset = torch.utils.data.Subset(dataset, range(subset_step))
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.dataset.collate_fn,
        )
    def get_offset_audio(self, audio_tensor, sample_rate):      
        """
            audio_tensor: [B, T]
            sample_rate: audio sample rate
            return: [B, T]
        """   
        shift_samples = int(sample_rate * self.shift_time / 1000)
        y_shifted = torch.roll(audio_tensor, shifts=shift_samples, dims=1)
        y_shifted[:, :shift_samples] = 0
        return y_shifted

    def get_same_id(self, gt_id, rec_id):
        """
            gt_id: [1, K, T/hop_length]
            rec_id: [1, K, T/hop_length]
            return: same id list
        """
        results = []
        for i in range(gt_id.shape[1]):
            same = (gt_id[0, i, :] == rec_id[0, i, :]).sum().item()
            results.append(same)
        return results

    def get_id(self, gt, rec):
        """
            gt: [B, T]
            rec: [B, T]
            return: gt_id: [1, K, T/hop_length] rec_id: [1, K, T/hop_length]
        """
        gt = gt.to(self.device)
        rec = rec.to(self.device)
        gt_id, _ = self.codec_encode(gt, length=None)
        gt_id = gt_id.movedim(-1, -2)
        rec_id, _ = self.codec_encode(rec, length=None)
        rec_id = rec_id.movedim(-1, -2)
        return gt_id, rec_id

    def multi_round_reconstruction(self, gt_audio_clone, gt_audio, max_length, rec_audio_dict, gt_audio_dict):
        """
            gt_audio_clone: [B, T]
            gt_audio: [B, T]
            max_length: max length of the gt_audio_clone
            rec_audio_dict: {round_1: [], round_2: [], round_3: [],...}
            gt_audio_dict: {round_1: [], round_2: [], round_3: [],...}
        """
        for i in range(10):
            rec_audio = self.codec_reconstruct(gt_audio)[:, :max_length]    # shape [B, T]
            min_length = min(gt_audio.shape[-1], rec_audio.shape[-1])
            for idx, rec_a in enumerate(rec_audio):
                rec_audio_dict[f"round_{i + 1}"].append(rec_a[:min_length].unsqueeze(0).to("cpu"))
                gt_audio_dict[f"round_{i + 1}"].append(gt_audio_clone[idx][:min_length].unsqueeze(0).to("cpu"))
                
            gt_audio = rec_audio
            max_length = gt_audio.shape[-1]

        return rec_audio_dict, gt_audio_dict

    def data_process(self, same_id_dict_round):
        """
            same_id_dict_round: {round_1: [codebook_1:[] codebook_2:[],...], round_2: [],...}
            return: result_codebook_same_id: {codebook_1: [round_1:[],round_2:[]...], codebook_2: [],...}
        """
        round_list = [f'round_{i+1}' for i in range(10)]
        avg_same_id_dict = {} # {round_1: [avg_codebook_1, avg_codebook_2, ...], round_2: [], ...}
        for i in range(10): # round
            avg_same_id_dict[round_list[i]] = []

        for key, value_same_id_dict in same_id_dict_round.items(): # round
            for _, value_same_id in value_same_id_dict.items(): # codebook
                if value_same_id != []:
                    numeric_values = [float(val.rstrip('%')) for val in value_same_id]
                    average = sum(numeric_values) / len(numeric_values)
                    average_with_percentage = "{:.2f}%".format(average)
                    avg_same_id_dict[key].append(average_with_percentage)
        
        result_codebook_same_id = {}
        for i in range(self.num_codebooks):
            codebook_key = f'codebook{i+1}'
            result_codebook_same_id[codebook_key] = []
            for round_values in avg_same_id_dict.values():
                result_codebook_same_id[codebook_key].append(round_values[i])

        return result_codebook_same_id

    @torch.inference_mode
    def evaluate(self, task):
        if task == "MRC":
            gt_audio_dict = {}
            rec_audio_dict = {}
            for i in range(10):
                gt_audio_dict[f"round_{i + 1}"] = []    # {round_1: [], round_2: [], round_3: [], ...}
                rec_audio_dict[f"round_{i + 1}"] = []   # {round_1: [], round_2: [], round_3: [], ...}

            print(f"dataset length: {len(self.dataloader)}, now start to reconstruct")
            gt_audio_lengths = []
            for batch in tqdm(self.dataloader, desc="reconstruct audio"):
                gt_audio_test = batch["audio"].clone().to(self.device)  # shape [B, T]
                gt_audio_length = batch["audio_length"] 
                gt_audio_lengths.append(gt_audio_length)
                max_length = gt_audio_test.shape[-1]
                gt_audio = batch["audio"].to(self.device)   # shape [B, T]

                rec_audio_dict, gt_audio_dict = self.multi_round_reconstruction(
                    gt_audio_clone=gt_audio_test,
                    gt_audio=gt_audio,
                    max_length=max_length,
                    rec_audio_dict=rec_audio_dict,
                    gt_audio_dict=gt_audio_dict,
                )
            torch.cuda.empty_cache()
            print(f"Reconstruct done! Next, carry out the MRC task.")

            same_id_dict_round = {}     # {round_1: [codebook_1:[] codebook_2:[], ...], round_2: [], ...}
            for i in range(10): # round
                same_id_dict_round[f"round_{i + 1}"] = {}
                for j in range(self.num_codebooks): # codebook
                    same_id_dict_round[f"round_{i + 1}"][f"codebook_{j + 1}"] = []

            all_lengths = [length for sublist in gt_audio_lengths for length in sublist]
            for key, value in gt_audio_dict.items(): # round
                gt_audio_list = value
                rec_audio_list = rec_audio_dict[key]
                for i, (gt, rec) in enumerate(zip(gt_audio_list, rec_audio_list)): # bs
                    if i < len(all_lengths):
                        gt_audio_length = all_lengths[i]
                        gt = gt[:, :gt_audio_length]
                        rec = rec[:, :gt_audio_length]
                        gt_id, rec_id = self.get_id(gt, rec)   
                        same_id_result_list = self.get_same_id(gt_id, rec_id)    
                        for idx,same_id in enumerate(same_id_result_list): # codebook
                            same_id_dict_round[key][f"codebook_{idx + 1}"].append("{:.2f}%".format((same_id / gt_id.shape[2]) * 100))
            avg_same_id_dict = self.data_process(same_id_dict_round)
            plot_mrc_avg_same_id(avg_same_id_dict, self.codec_name, self.task)
            return avg_same_id_dict
        
        elif task == "OS":
            same_id_sums = [0] * self.num_codebooks
            total_samples = 0
            for batch in tqdm(self.dataloader, desc="dataloader audio"):
                gt_audio_test = batch["audio"].to(self.device)  # shape [B, T]
                all_gt_audio_lengths = batch["audio_length"]
                for i, gt_audio in enumerate(gt_audio_test):
                    gt_audio = gt_audio_test[i].to(self.device)
                    gt_audio_length = all_gt_audio_lengths[i]
                    gt_audio = gt_audio.unsqueeze(0)
                    gt_audio = gt_audio[:, :gt_audio_length]
                    os_audio = self.get_offset_audio(gt_audio, self.sample_rate)
                    gt_id, rec_id = self.get_id(gt_audio, os_audio)
                    same_id_results = self.get_same_id(gt_id, rec_id)
                    total_samples += 1
                    for j in range(self.num_codebooks):
                        same_id_sums[j] += (same_id_results[j] / gt_id.shape[2])
            percent_same_id_avg_list = [f"{(val / total_samples) * 100:.2f}%" for val in same_id_sums]
            plot_os_avg_same_id(percent_same_id_avg_list, self.num_codebooks, self.codec_name, self.task)
            return f"codebook same id: {percent_same_id_avg_list}"


def main():
    seed_all(666)
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="speechtokenizer")
    parser.add_argument(
        "--model_ckpt_dir",
        type=str,
        default="/sdb/model_weight/codec_evaluation/codec_ckpt/speechtokenizer",
    )
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--need_resample", type=bool, default=True)
    parser.add_argument("--task", type=str, default="MRC")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shift_time", type=int, default=2)
    parser.add_argument("--subset_step", type=int, default=1200)
    parser.add_argument(
        "--dataset_audio_dir",
        type=str,
        default="/sdb/data1/speech/24kHz/LibriTTS/test-other",
    )
    parser.add_argument("--use_vocos", type=bool, default=False)
    parser.add_argument("--vocos_ckpt_dir", type=Optional[str], default=None)
    args = parser.parse_args()

    codec_eval = IDSensitiveEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        device=args.device,
        sample_rate=args.sample_rate,
        num_codebooks=args.num_codebooks,
        need_resample=args.need_resample,
        dataset_audio_dir=args.dataset_audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_vocos=args.use_vocos,
        vocos_ckpt_dir=args.vocos_ckpt_dir,
        shift_time=args.shift_time,
        subset_step=args.subset_step,
        task=args.task
    )
    result = codec_eval.evaluate(args.task)
    print(result)


if __name__ == "__main__":
    main()
