import pandas as pd
import json
from config import Config

emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name
data_path = Config.datapath
keep_neutral = Config.keep_neutral

def get_emotion_text_from_json(data):
    df = pd.DataFrame(columns=["text", emotion_colm_name])
    for x in data["episodes"]:
        for i in x["scenes"]:
            for j in i["utterances"]:

                dict = {"text": j["transcript"], emotion_colm_name: j[emotion_colm_name]}
                df = pd.concat([df, pd.DataFrame([dict])], ignore_index=True)
    return df

def load_emotion_detection_master_data():
    main_path = data_path + "emotion-detection-master/emotion-detection-master"
    dev_path = main_path + "/json/emotion-detection-dev.json"
    trn_path = main_path + "/json/emotion-detection-trn.json"
    tst_path = main_path + "/json/emotion-detection-tst.json"


    df_dev = pd.DataFrame(columns=[sentence_colmn_name, emotion_colm_name])


    with open(dev_path, 'r') as f:
        dev_data=json.load(f)
    with open(trn_path, 'r') as f:
        trn_data=json.load(f)
    with open(tst_path, 'r') as f:
        tst_data=json.load(f)

    df_dev = get_emotion_text_from_json(dev_data)
    df_trn = get_emotion_text_from_json(trn_data)
    df_tst = get_emotion_text_from_json(tst_data)

    df = pd.concat([df_dev, df_trn, df_tst], ignore_index=True)

    if not keep_neutral:
        df = df[df[emotion_colm_name] != "Neutral"]

    df.loc[:, emotion_colm_name] = df.loc[:, emotion_colm_name].replace("Joyful", "Happy")
    df.loc[:, emotion_colm_name] = df.loc[:, emotion_colm_name].replace("Peaceful", "Happy")
    df.loc[:, emotion_colm_name] = df.loc[:, emotion_colm_name].replace("Powerful", "Neutral")

    if not keep_neutral:
        df = df[df[emotion_colm_name] != "Neutral"]
    else:
        df.loc[:, emotion_colm_name] = df.loc[:, emotion_colm_name].replace("Neutral", "neutral")



    df.rename(columns={"text": sentence_colmn_name}, inplace=True)
    return df