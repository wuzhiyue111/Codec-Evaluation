from codec_evaluation.init_codecs import init_codec
import torchaudio
import random
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from codec_evaluation.probe.dataset.LibriTTS_dataset.libritts_ctc import (
    LibriTTS_ctc_dataset,
)
from codec_evaluation.id_sensitive.utils import (
    longest_common_substring_strict,
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


seed_all(666)

class IDSensitiveEvaluation:
    def __init__(
        self, 
        codec_name: str,
        model_ckpt_dir: str,
        sample_rate: int,
        num_codebooks: int,
        need_resample: bool,
        dataset_audio_dir: str,
        device: str ,  
        batch_size: int = 32,
        num_workers: int = 8,
        
        **kwargs,
    ):
        """
        codec_name: codec model name
        model_ckpt_dir: codec model checkpoint directory
        sample_rate: audio sample rate
        mode_encode: codec mode for encoding
        mode_reconstruct: codec mode for reconstruction
        num_codebooks: number of codebooks
        need_resample: boolean, whether to resample the audio after decoding
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.codec_encode = init_codec(
            modelname=codec_name,
            mode="encode",
            sample_rate=sample_rate,
            device=self.device,  # 传递设备参数
            num_codebooks=num_codebooks,
            need_resample=need_resample,  
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        ).to(self.device)  # 将模型移动到设备上

        self.codec_reconstruct = init_codec(
            modelname=codec_name,
            mode="reconstruct",
            sample_rate=sample_rate,
            device=self.device,  # 传递设备参数
            num_codebooks=num_codebooks,
            need_resample=need_resample,  
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        ).to(self.device)  # 将模型移动到设备上

        self.codec_name = codec_name
        self.codec_sample_rate = self.codec_encode.orig_sample_rate
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = device

        dataset = LibriTTS_ctc_dataset(audio_dir=dataset_audio_dir)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )
    
    def get_offset_2ms_audio(self, audio_tensor, sample_rate):
        y = audio_tensor.cpu().numpy()
        shift_samples = int(sample_rate * 2 / 1000)  
        y_shifted = np.roll(y, shift_samples, axis=1)# 向前移动（右移），前面填充 0
        y_shifted[:, :shift_samples] = 0  # 避免循环偏移
        y_shifted = torch.from_numpy(y_shifted).to(self.device)
        return y_shifted

    def get_same_id(self, id_1, id_2):
        results = []
        for i in range(id_1.shape[1]):
            same = (id_1[0, i, :] == id_2[0, i, :]).sum().item()
            results.append(same)
        result_dict = {"codebook same": results}
        return result_dict
 
    def get_lcs(self, emb1_id, emb2_id):
        lcs_lengths = []
        for i in range(emb1_id.shape[1]):  
            emb1_codebook = emb1_id[0, i, :].tolist()
            emb2_codebook = emb2_id[0, i, :].tolist()
            max_length = longest_common_substring_strict(emb1_codebook, emb2_codebook)
            lcs_lengths.append(max_length)
        return lcs_lengths
    
    def get_id(self, sig1, sig2):
        sig1 = sig1.to(self.device)
        sig2 = sig2.to(self.device)
        if self.codec_name == "wavtokenizer":
            id_1, _ = self.codec_encode(sig1, length=None)
            id_1 = id_1.movedim(-1, 0)
            id_2, _ = self.codec_encode(sig2, length=None)
            id_2 = id_2.movedim(-1, 0)
        else:
            id_1, _= self.codec_encode(sig1, length=None)
            id_1 = id_1.movedim(-1, -2)
            id_2, _ = self.codec_encode(sig2, length=None)
            id_2 = id_2.movedim(-1, -2)
        return id_1, id_2
    
    @torch.inference_mode
    def evaluate(self, task):
        all_results = []
        if task == "MRC":
            gt_audio_list = []
            rec_audio_dict = {}
            for i in range(10):
                rec_audio_dict[f'round_{i + 1}'] = []

            print(f"dataset length: {len(self.dataloader)}, now start to reconstruct")
            for batch in tqdm(self.dataloader, desc="reconstruct audio"):
                gt_audio_test = batch["audio"].clone()
                # resample to codec sample rate
                if self.codec_sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.sample_rate, self.codec_sample_rate
                    ).to(gt_audio_test.device)
                    gt_audio_test = resampler(gt_audio_test)

                gt_audio = batch["audio"].to(self.device)
                audio_lengths = batch["audio_length"]
                max_length = audio_lengths.max().item()

                if self.codec_sample_rate != self.sample_rate:
                    max_length = int(max_length * (self.codec_sample_rate / self.sample_rate))

                # 存储原始音频，仅需处理一次
                min_length = float('inf')
                for gt_audio_t in gt_audio_test:
                    min_length = min(min_length, gt_audio_t.shape[-1])
                for gt_audio_t in gt_audio_test:
                    gt_audio_list.append(gt_audio_t[:min_length].to("cpu"))

                # prevent the more than max_length
                for i in range(10):
                    rec_audio = self.codec_reconstruct(gt_audio)[:, :max_length]
                    # prevent the less than gt_audio
                    min_length = min(gt_audio_test.shape[-1], rec_audio.shape[-1])

                    # save to list
                    for rec_a in rec_audio:
                        rec_audio_dict[f'round_{i + 1}'].append(rec_a[:min_length].to("cpu"))

            torch.cuda.empty_cache()
            print(f"reconstruct done!")

            same_id_list = []
            lcs_list = []
            for i in range(len(gt_audio_list)):
                sig1 = gt_audio_list[i].to(self.device).unsqueeze(0)
                for round_num in range(1, 11):  # 重构了 10 轮
                    sig2 = rec_audio_dict[f'round_{round_num}'][i].to(self.device).unsqueeze(0)
                    id_1, id_2 = self.get_id(sig1, sig2)
                    same_id_results = self.get_same_id(id_1, id_2)
                    lcs_results = self.get_lcs(id_1, id_2)
                    same_id_list.append(same_id_results)
                    lcs_list.append(lcs_results)
            print("Same ID Results:", same_id_results)
            print("LCS Results:", lcs_results)
            # 绘制 Same ID Results 图表
            plt.figure(figsize=(10, 6))
            plt.plot(same_id_list, marker='o')
            plt.title('Same ID Results')
            plt.xlabel('Round')
            plt.ylabel('Same ID Value')
            plt.grid(True)
            plt.savefig('same_id_results.png')
            plt.close()

            # 绘制 LCS Results 图表
            plt.figure(figsize=(10, 6))
            plt.plot(lcs_list, marker='o')
            plt.title('LCS Results')
            plt.xlabel('Round')
            plt.ylabel('LCS Value')
            plt.grid(True)
            plt.savefig('lcs_results.png')
            plt.close()

        elif task == "OB2":
            for batch in tqdm(self.dataloader, desc="dataloader audio"):
                gt_audio_test = batch["audio"].clone()
                for i in range(len(gt_audio_test)):
                    sig1 = gt_audio_test[i].to(self.device)
                    sig1 = sig1.unsqueeze(0)
                    sig2 = self.get_offset_2ms_audio(sig1, self.sample_rate)
                    id_1, id_2 = self.get_id(sig1, sig2)
                    same_id_results = self.get_same_id(id_1, id_2)
                    lcs_results = self.get_lcs(id_1, id_2)
                print("Same ID Results:", same_id_results)
                print("LCS Results:", lcs_results)        

        return all_results

if __name__ == "__main__":
    """
        if task == "MRC":
            test sample_rate = codec's sample_rate
        elif task == "OB2":
            test sample_rate = example.wav's sample_rate
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="dac")
    parser.add_argument("--model_ckpt_dir", type=str, default="/sdb/model_weight/codec_evaluation/codec_ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--need_resample", type=bool, default=False)
    parser.add_argument("--task", type=str, default="MRC")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset_audio_dir", type=str, default="/sdb/data1/speech/24kHz/LibriTTS/test-other")
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
    )
    result = codec_eval.evaluate(args.task)
    print(result)    