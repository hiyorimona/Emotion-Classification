#%% IMPORT
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import numpy as np
import pandas as pd
from tqdm import tqdm
import whisper
import re
import os



class DataPreprocessing:


    def extract_number(self, f):
        s = re.findall("\d+", f)
        return (int(s[0]) if s else -1, f)

    def extract_number_2(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None

    def mov_to_mp3_audio(self, input_file, output_file):
        video_clip = VideoFileClip(input_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_file)
        audio_clip.close()
        video_clip.close()

    def filter_survivor_df(self, df, episode):
        df = df[df["Episode name"] == episode]
        start_times = df["Start Time (seconds)"].unique()
        end_times = df["End Time (seconds)"].unique()

        return start_times, end_times

    def split_mp3(self, input_file, output_prefix, start_times, end_times):
        # Load the mp3 file
        audio = AudioSegment.from_mp3(input_file)

        end_times = np.concatenate((start_times[1:], end_times[-1].reshape(1)))

        # Convert start and end times to milliseconds
        start_times_ms = [int(start_time * 1000) for start_time in start_times]
        end_times_ms = [int(end_time * 1000) for end_time in end_times]

        # Save each segment as a separate mp3 file
        for i, (start_time, end_time) in enumerate(zip(start_times_ms, end_times_ms)):
            segment = audio[start_time:end_time]
            output_file = f"{output_prefix}_{i+1}.mp3"
            segment.export(output_file, format="mp3")


    def transcripe_translate(self, path, sorted_files, model):
        text_dict = {"text": []}
        print("Transcribing and translating audio files...")
        i = 0
        for file in sorted_files:
            i += 1
            print(f"Transcribing and translating audio file {i}/{len(sorted_files)}")
            transcription = model.transcribe(f"{path}{file}",  language="nl", task="translate")
            text_dict["text"].append(transcription["text"])
        df = pd.DataFrame(text_dict)
        return df



    def run(self, structured_df, files, path, model):
        i = 5
        for file in tqdm(files):
            full_path = f"{path}/{files}"
            if os.path.isdir(full_path):
                for video in os.listdir(full_path):
                    if video.endswith(".mov"):
                        i += 1
                        full_path_video = f"{full_path}/{video}"
                        # convert video to audio
                        audio_video = video.replace(".mov", ".mp3")
                        full_path_save_audio = f"{full_path}/{audio_video}"
                        self.mov_to_mp3_audio(full_path_video, full_path_save_audio)

                        # split up segments
                        start_times, end_times = self.filter_survivor_df(structured_df, i)
                        if not os.path.exists(full_path + "/split_up"):
                            os.makedirs(full_path + "/split_up")
                        output_split_audio = full_path + "/split_up/"

                        self.split_mp3(full_path_save_audio, output_split_audio+"segment", start_times, end_times)
                        files = os.listdir(output_split_audio)


                        sorted_files = sorted((file for file in files if file.endswith(".mp3")), key=self.extract_number)

                        # get transcription and translation per segment
                        df = self.transcripe_translate(output_split_audio, sorted_files, model)
                        df.to_csv(f"{full_path}/transcriptions.csv", index=False)



if __name__ == "__main__":
    processor = DataPreprocessing()
    path = "episodes"
    model = whisper.load_model("large")
    structure_df = pd.read_csv(f"{path}/Robinson22_structure.csv")
    files = os.listdir(path)

    processor.run(structure_df, files, path, model)
