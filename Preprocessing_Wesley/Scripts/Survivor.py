import pandas as pd
from config import Config


emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name
main_path = Config.datapath

data_path = main_path + "Survivor"


happiness_name = Config.happy
sadness_name = Config.sad
mad_name = Config.mad
disgusted_name = Config.disgusted
surprised_name = Config.surpised
fear_name = Config.scared

keep_neutral = Config.keep_neutral


def load_survivor_data():
    season_42 = data_path + "/Season_42.csv"
    season_43 = data_path + "/Season_43.csv"
    season_44 = data_path + "/Season_44.csv"
    

    df = pd.read_csv(season_44)
    df2 = pd.read_csv(season_43)
    df3 = pd.read_csv(season_42)


    df = pd.concat([df, df2, df3])

    df["emotion"] = df["emotion"].replace("Neutral", "neutral")
    df["emotion"] = df["emotion"].replace("Happiness", "happiness")
    df["emotion"] = df["emotion"].replace("Sadness", "sadness")
    df["emotion"] = df["emotion"].replace("Anger", "anger")
    df["emotion"] = df["emotion"].replace("Surprise", "surprise")
    df["emotion"] = df["emotion"].replace("Fear", "fear")
    df["emotion"] = df["emotion"].replace("Disgust", "disgust")
    df["emotion"] = df["emotion"].replace("Confusion", "surprise")
    df["emotion"] = df["emotion"].replace("Disappointment", "sadness")
    df["emotion"] = df["emotion"].replace("guilt", "sadness")
    df["emotion"] = df["emotion"].replace("discomfort", "sadness")
    df["emotion"] = df["emotion"].replace("Hopefulness", "happiness")
    df["emotion"] = df["emotion"].replace("confusion", "surprise")
    emotions = ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral']
    df = df[df['emotion'].isin(emotions)]

    if keep_neutral == False:
        df = df[df["emotion"] != "neutral"]

    df.rename(columns={"Text": sentence_colmn_name}, inplace=True)
    df.rename(columns={"emotion": emotion_colm_name}, inplace=True)

    return df



