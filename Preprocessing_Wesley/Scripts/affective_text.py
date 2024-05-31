import pandas as pd
from config import Config

main_path = Config.datapath
emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name

affectivetext_path = main_path + "AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml"
affectivetext_test_emotions_path = main_path + "AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold"


def prepare_affective_text():
    with open(affectivetext_path) as file:
        data = file.read()

    data = data.replace("&", "&amp;")

    with open(affectivetext_path, "w") as file:
        file.write(data)

    affective_text_df = pd.read_xml(affectivetext_path)

    with open(affectivetext_test_emotions_path) as file:
        data = file.read()

    lines = data.split('\n')
    df_emotions = pd.DataFrame(line.split(' ') for line in lines)
    df_emotions.columns = ['id', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    df_emotions = df_emotions.dropna()

    affective_df = pd.concat([affective_text_df, df_emotions], axis=1)

    affective_df = affective_df.dropna()

    affective_df = affective_df.astype({'anger': 'int', 'disgust': 'int', 'fear': 'int', 'joy': 'int', 'sadness': 'int', 'surprise': 'int'})

    affective_df[emotion_colm_name] = affective_df[["anger", "disgust", "fear", "joy", "sadness", "surprise"]].iloc[:, 1:].idxmax(axis=1)

    affective_df = affective_df[["instance", emotion_colm_name]]
    affective_df.rename(columns={"instance": sentence_colmn_name}, inplace=True)

    affective_df.loc[:, emotion_colm_name] = affective_df.loc[:, emotion_colm_name].replace("joy", "Happy")
    affective_df.loc[:, emotion_colm_name] = affective_df.loc[:, emotion_colm_name].replace("fear", "Scared")
    affective_df.loc[:, emotion_colm_name] = affective_df.loc[:, emotion_colm_name].replace("sadness", "Sad")
    affective_df.loc[:, emotion_colm_name] = affective_df.loc[:, emotion_colm_name].replace("surprise", "Surprised")
    affective_df.loc[:, emotion_colm_name] = affective_df.loc[:, emotion_colm_name].replace("disgust", "Disgusted")

    return affective_df
