import pandas as pd
from config import Config

data_path = Config.datapath
emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name

keep_neutral = Config.keep_neutral

def load_meld_master_data():
    main_path = data_path + "MELD-master/MELD-master/data/MELD"

    df_dev_sent_path = main_path + "/dev_sent_emo.csv"
    df_test_sent_path = main_path + "/test_sent_emo.csv"
    df_trian_sent_path = main_path + "/train_sent_emo.csv"

    df_dev_sent = pd.read_csv(df_dev_sent_path)
    df_test_sent = pd.read_csv(df_test_sent_path)
    df_train_sent = pd.read_csv(df_trian_sent_path)

    df_dev_sent = df_dev_sent[["Utterance", "Emotion"]]
    df_test_sent = df_test_sent[["Utterance", "Emotion"]]
    df_train_sent = df_train_sent[["Utterance", "Emotion"]]

    df = pd.concat([df_dev_sent, df_test_sent, df_train_sent])

    df.rename(columns={"Emotion": "emotion"}, inplace=True)

    if not keep_neutral:
        df = df[df["emotion"] != "neutral"]
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("sadness", "Sad")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("surprise", "Surprised")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("joy", "Happy")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("anger", "Mad")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("disgust", "Disgusted")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("fear", "Scared")


    df.rename(columns={"Utterance": sentence_colmn_name, "emotion":emotion_colm_name}, inplace=True)


    return df