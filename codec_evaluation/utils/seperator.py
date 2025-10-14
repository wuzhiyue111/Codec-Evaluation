# https://github.com/tencent-ailab/SongGeneration/blob/main/tools/gradio/separator.py
import torchaudio
import os
import torch
from codec_evaluation.utils.demucs.models.pretrained import get_model_from_yaml
from codec_evaluation.utils.demucs.models.apply import apply_model


class Separator(torch.nn.Module):
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        super().__init__()
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model

    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        else:
            a = torch.cat([a, a], -1)
        return a[:, 0:48000*10]

    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio

    def separate_from_mix(self, mix_audio, sample_rate=24000):
        """
        mix_audio: Tensor, (B, T)
        sample_rate: input & output sample rate
        return: mix_audio, vocal_audio, bgm_audio, Tensor, (B, T)
        """
        mix_audio = mix_audio.to(self.device)
        model_sr = getattr(self.demucs_model, 'samplerate', sample_rate)
        target_channels = getattr(self.demucs_model, 'audio_channels', 1)

        # to (B, C, T)
        mix = mix_audio.unsqueeze(1)
        # match model sample rate
        if sample_rate != model_sr:
            mix = torchaudio.functional.resample(mix, sample_rate, model_sr)
        # match model channels
        if mix.shape[1] != target_channels:
            mix = mix.repeat(1, target_channels, 1)

        # normalize exactly as in BagOfModels.separate
        ref = mix.mean(1)  # (B, T)
        mix = mix - ref.mean(-1, keepdim=True).unsqueeze(1)
        mix = mix / ref.std(-1, keepdim=True).unsqueeze(1)

        with torch.no_grad():
            sources = apply_model(
                self.demucs_model,
                mix,
                shifts=1,
                split=True,
                overlap=0.25,
                transition_power=1.0,
                progress=True,
                device=self.device,
                num_workers=0,
                segment=None,
            )

        # denormalize exactly as in BagOfModels.separate
        sources = sources * ref.std(-1, keepdim=True).view(-1, 1, 1, 1) + ref.mean(-1, keepdim=True).view(-1, 1, 1, 1)

        # pick vocals and downmix to mono (B, T)
        v_idx = 3
        vocal_audio = sources[:, v_idx].mean(1)

        # resample back to input sample rate
        if sample_rate != model_sr:
            vocal_audio = torchaudio.functional.resample(vocal_audio, model_sr, sample_rate)

        # bgm = mix - vocals at input sr
        bgm_audio = mix_audio - vocal_audio
        return mix_audio, vocal_audio, bgm_audio
    
if __name__ == "__main__":
    separator = Separator(dm_model_path='/codec_evaluation/utils/demucs/ckpt/htdemucs.pth', 
                          dm_config_path='/codec_evaluation/utils/demucs/ckpt/htdemucs.yaml', 
                          gpu_id=0)
    audio, sr = torchaudio.load("王力宏 _ 就是现在 _ 20150123.flac")
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    if audio.shape[0] != 1:
        audio = audio.mean(dim=0).unsqueeze(0) # shape (1, T)

    mix_audio, vocal_audio, bgm_audio = separator.separate_from_mix(audio, sample_rate=24000)