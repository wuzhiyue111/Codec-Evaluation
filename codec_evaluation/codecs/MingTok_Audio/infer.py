import torch
import torchaudio
import codec_evaluation
import sys
root_path = codec_evaluation.__path__[0]
sys.path.append(root_path)
from audio_tokenizer.modeling_audio_vae import AudioVAE

model = AudioVAE.from_pretrained("/sdb/model_weight/codec_evaluation/codec_ckpt/mingtok_audio")
model = model.cuda()
model.eval()

waveform, sr = torchaudio.load('codec_evaluation/codecs/example.wav', backend='soundfile')
sample = {'waveform': waveform.cuda(), 'waveform_length': torch.tensor([waveform.size(-1)]).cuda()}

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent, frame_num = model.encode_latent(**sample)
        output_waveform = model.decode(latent)

torchaudio.save('./mingtokaudio_reconstruction.wav', output_waveform.cpu()[0], sample_rate=16000)