import pandas as pd
import json
from config import Config

emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name
main_path = Config.datapath

data_path = main_path + "data.jsonl/data.jsonl"


def load_carer_data():
    with open(data_path, "r") as file:
        data = file.readlines()

    parsed_data = [json.loads(line) for line in data if line]

    df = pd.DataFrame(parsed_data)

    df = df.rename(columns={'label': emotion_colm_name})
    df = df.rename(columns={'text': sentence_colmn_name})

    df[emotion_colm_name] = df[emotion_colm_name].astype(str)
    df[emotion_colm_name] = df[emotion_colm_name].replace('0', "Sad")
    df[emotion_colm_name] = df[emotion_colm_name].replace('1', "Happy")
    df[emotion_colm_name] = df[emotion_colm_name].replace('2', "Happy")
    df[emotion_colm_name] = df[emotion_colm_name].replace('3', "Mad")
    df[emotion_colm_name] = df[emotion_colm_name].replace('4', "Scared")
    df[emotion_colm_name] = df[emotion_colm_name].replace('5', "Surprised")

    return df
