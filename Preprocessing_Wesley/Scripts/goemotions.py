import pandas as pd
from config import Config

data_path = Config.datapath
emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name

main_path = data_path + "full_dataset"
keep_neutral = Config.keep_neutral

prioritize_happy = Config.priotize_happy


def load_goemotion_data():
    df_geomotion_1 = pd.read_csv(main_path + "/goemotions_1.csv")
    df_geomotion_2 = pd.read_csv(main_path + "/goemotions_2.csv")
    df_geomotion_3 = pd.read_csv(main_path + "/goemotions_3.csv")

    # Mapping of GoEmotions to Ekman's emotions
    emotion_mapping = {
        'admiration': 'happiness',
        'amusement': 'happiness',
        'anger': 'anger',
        'annoyance': 'anger',
        'approval': 'happiness',
        'caring': 'happiness',
        'confusion': 'surprise',
        'curiosity': 'surprise',
        'desire': 'happiness',
        'disappointment': 'sadness',
        'disapproval': 'disgust',
        'disgust': 'disgust',
        'embarrassment': 'sadness',
        'excitement': 'happiness',
        'fear': 'fear',
        'gratitude': 'happiness',
        'grief': 'sadness',
        'joy': 'happiness',
        'love': 'happiness',
        'nervousness': 'fear',
        'optimism': 'happiness',
        'pride': 'happiness',
        'realization': 'surprise',
        'relief': 'happiness',
        'remorse': 'sadness',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }

    if prioritize_happy:
        def find_emotion(row):
            # Define a list of emotions that you consider as forms of happiness
            happiness_forms = ['joy', 'amusement', 'gratitude', 'love', 'optimism', 'pride', 'relief', 'admiration', 'approval', 'desire']
            for happy_emotion in happiness_forms:
                if row[happy_emotion] == 1:
                    return 'happiness'
            for emotion, ekman_emotion in emotion_mapping.items():
                if row[emotion] == 1:
                    return ekman_emotion
            return 'neutral'
    else:
        def find_emotion(row):
            for emotion, ekman_emotion in emotion_mapping.items():
                if row[emotion] == 1:
                    return ekman_emotion
            return 'neutral'

    # Apply the function to each row
    df_geomotion_1[emotion_colm_name] = df_geomotion_1.apply(find_emotion, axis=1)
    df_geomotion_2[emotion_colm_name] = df_geomotion_2.apply(find_emotion, axis=1)
    df_geomotion_3[emotion_colm_name] = df_geomotion_3.apply(find_emotion, axis=1)

    # Keep only the 'text' and emotion_colm_name columns
    df_geomotion_1 = df_geomotion_1[['text', 'emotion']]
    df_geomotion_2 = df_geomotion_2[['text', 'emotion']]
    df_geomotion_3 = df_geomotion_3[['text', 'emotion']]

    df = pd.concat([df_geomotion_1, df_geomotion_2, df_geomotion_3])

    if not keep_neutral:
        df = df[df["emotion"] != "neutral"]
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("sadness", "Sad")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("surprise", "Surprised")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("happiness", "Happy")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("anger", "Mad")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("disgust", "Disgusted")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("fear", "Scared")


    df.rename(columns={"text": sentence_colmn_name, "emotion":emotion_colm_name}, inplace=True)


    return df



