import pandas as pd
import os
from config import Config

data_path = Config.datapath
emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name
suprised_positive = Config.suprised_positive
suprised_negative = Config.suprised_negative

keep_neutral = Config.keep_neutral

def process_fairy_tail(df):
    df = df[[1, 3]]
    df.columns = ['emotion', 'text']
    df.loc[:, "emotion"] = df.loc[:, "emotion"].apply(lambda x: x.split(':')[0])

    if keep_neutral:
        df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("N", "Neutral")
    else:
        df = df[df["emotion"] != "N"]
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("Sa", "Sad")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("H", "Happy")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("D", "Disgusted")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("A", "Mad")
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("Su+", suprised_positive)
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("Su-", suprised_negative)
    df.loc[:, "emotion"] = df.loc[:, "emotion"].replace("F", "Scared")

    return df


def load_fairy_tail_data():
    main_path_grimms = data_path + "Grimms/Grimms/emmood/"
    main_path_potter = data_path + "Potter/Potter/emmood/"
    main_path_hcandersen = data_path + "HCAndersen/HCAndersen/emmood/"


    df_grims = pd.DataFrame(columns=['emotion', 'text'])
    for file in os.listdir(main_path_grimms):
        df_temp = pd.read_csv(main_path_grimms + file, delimiter='\t', header=None)
        df_pros = process_fairy_tail(df_temp)

        df_grims = pd.concat([df_grims, df_pros])


    df_potter = pd.DataFrame(columns=['emotion', 'text'])
    for file in os.listdir(main_path_potter):
        df_temp = pd.read_csv(main_path_potter + file, delimiter='\t', header=None)
        df_pros = process_fairy_tail(df_temp)

        df_potter = pd.concat([df_potter, df_pros])


    df_hcandersen = pd.DataFrame(columns=['emotion', 'text'])
    for file in os.listdir(main_path_hcandersen):
        df_temp = pd.read_csv(main_path_hcandersen + file, delimiter='\t', header=None)
        df_pros = process_fairy_tail(df_temp)

        df_hcandersen = pd.concat([df_hcandersen, df_pros])

    
    df = pd.concat([df_grims, df_potter, df_hcandersen])

    df = df[['text', 'emotion']]
    df.rename(columns={"text": sentence_colmn_name, "emotion":emotion_colm_name}, inplace=True)

    return df