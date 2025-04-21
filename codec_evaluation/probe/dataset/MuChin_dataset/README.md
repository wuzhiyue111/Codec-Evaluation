# How to Use MuChin_dataset in CTC Task?
   
## Preprocess dataset: One lyric corresponds to one audio
     
1.Download MuChin_dataset

- More details in https://github.com/CarlWangChina/MuChin


2.Switch the working directory  

- `cd Codec-Evaluation/codec_evaluation/probe/dataset/Muchin_dataset`


3.Run preprocess_muchin.py
- `python preprocess_muchin.py --audio_directory /Your/path/to/muchin/muchin-audio --lyric_directory /Your/path/to/muchin/dataset/bm-data --output_audio_directory /Your/path/to/save/split/audio --output_json_file /Your/path/to/save/json_file.json`
