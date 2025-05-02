import os
import json
from pydub import AudioSegment


def parse_raw_lyric(raw_lyric_path):
    """
    解析 raw_lyric 文件，返回时间戳和歌词列表
    """
    timestamps = []
    lyrics = []
    with open(raw_lyric_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '[' in line and ']' in line:
                timestamp_str = line[line.find('[') + 1:line.find(']')]
                try:
                    minutes, seconds = map(float, timestamp_str.split(':'))
                    timestamp = minutes * 60 + seconds
                    timestamps.append(timestamp)
                    lyric = line[line.find(']') + 1:].strip()
                    lyrics.append(lyric)
                except ValueError:
                    continue
    return timestamps, lyrics


def split_audio(audio_path, raw_lyric_path, output_audio_dir):
    """
    根据 raw_lyric 文件的时间戳切分音频，并返回切分信息
    """
    audio = AudioSegment.from_file(audio_path)
    timestamps, lyrics = parse_raw_lyric(raw_lyric_path)
    audio_filename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_filename)[0]
    output_info = []

    for i in range(len(timestamps) - 1):
        start_time = timestamps[i] * 1000
        end_time = timestamps[i + 1] * 1000
        duration = (end_time - start_time) / 1000
        if duration <= 20:
            segment = audio[start_time:end_time]
            segment_filename = f"{audio_name}_split{i}.wav"
            segment_path = os.path.join(output_audio_dir, segment_filename)
            segment.export(segment_path, format='wav')
            info = {
                "filename": segment_filename,
                "lyric": lyrics[i],
                "start_time": timestamps[i],
                "end_time": timestamps[i + 1],
                "duration": duration
            }
            output_info.append(info)

    return output_info


def process_audio_directory(audio_dir, lyric_dir, output_audio_dir, output_json_path):
    """
    处理指定目录下的所有音频文件
    """
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)

    all_output_info = []
    for root, dirs, files in os.walk(audio_dir):
        for audio_file in files:
            audio_path = os.path.join(root, audio_file)
            audio_name = os.path.splitext(audio_file)[0]
            raw_lyric_path = os.path.join(lyric_dir, audio_name, 'raw_lyric')
            print("audio_path:", audio_path)
            print("audio_name:", audio_name)
            print("raw_lyric_path:", raw_lyric_path)
            if os.path.exists(raw_lyric_path):
                output_info = split_audio(audio_path, raw_lyric_path, output_audio_dir)
                all_output_info.extend(output_info)

    # 按文件名排序
    all_output_info.sort(key=lambda x: x["filename"])

    # 保存切分信息到 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_output_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    audio_directory = "/sdb/data1/benchmark/muchin/muchin-audio"  # 音频文件所在路径
    lyric_directory = "/sdb/data1/benchmark/muchin/dataset/bm-data"  # raw_lyric 文件所在路径
    output_audio_directory = "/sdb/data1/benchmark/muchin/muchin_split"  # 切分后音频保存路径
    output_json_file = "/sdb/data1/benchmark/muchin/muchin_ctc.json"  # 保存切分信息的 JSON 文件路径

    process_audio_directory(audio_directory, lyric_directory, output_audio_directory, output_json_file)