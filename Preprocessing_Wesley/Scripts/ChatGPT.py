import pandas as pd
from config import Config

emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name
main_path = Config.datapath

data_path = main_path + "ChatGPT/chat_gpt_generated2.csv"
data_path3 = main_path + "ChatGPT/chat_gpt_generated3.csv"
data_path_angry = main_path + "ChatGPT/chat_gpt_generated_angry.csv"
data_path_disgusted2 = main_path + "ChatGPT/chat_gpt_generated_disgust_2.csv"
data_path_disgusted4 = main_path + "ChatGPT/chat_gpt_generated_disgust_4.csv"

happiness_name = Config.happy
sadness_name = Config.sad
mad_name = Config.mad
disgusted_name = Config.disgusted
surprised_name = Config.surprised
fear_name = Config.scared

def load_gpt_data():
    df = pd.read_csv(data_path)
    df = df.rename(columns={'emotion': emotion_colm_name})
    df = df.rename(columns={'text': sentence_colmn_name})

    df2 = pd.read_csv(data_path3)
    df2 = df2.rename(columns={'emotion': emotion_colm_name})
    df2 = df2.rename(columns={'sentence': sentence_colmn_name})

    df_angry = pd.read_csv(data_path_angry)
    df_angry = df_angry.rename(columns={'emotion': emotion_colm_name})
    df_angry = df_angry.rename(columns={'sentence': sentence_colmn_name})
    df_angry[emotion_colm_name] = mad_name

    df_disgust2 = pd.read_csv(data_path_disgusted2)
    df_disgust2 = df_disgust2.rename(columns={'emotion': emotion_colm_name})
    df_disgust2 = df_disgust2.rename(columns={'sentence': sentence_colmn_name})
    df_disgust2[emotion_colm_name] = disgusted_name

    df_disgust4 = pd.read_csv(data_path_disgusted4)
    df_disgust4 = df_disgust4.rename(columns={'emotion': emotion_colm_name})
    df_disgust4 = df_disgust4.rename(columns={'sentence': sentence_colmn_name})
    df_disgust4[emotion_colm_name] = disgusted_name

    df2[emotion_colm_name] = disgusted_name

    df = pd.concat([df, df2, df_angry, df_disgust2, df_disgust4], ignore_index=True)

    df[emotion_colm_name] = df[emotion_colm_name].astype(str)
    df[emotion_colm_name] = df[emotion_colm_name].replace("happiness", happiness_name)
    df[emotion_colm_name] = df[emotion_colm_name].replace("sadness", sadness_name)
    df[emotion_colm_name] = df[emotion_colm_name].replace("anger", mad_name)
    df[emotion_colm_name] = df[emotion_colm_name].replace("disgust", disgusted_name)
    df[emotion_colm_name] = df[emotion_colm_name].replace("surprise", surprised_name)
    df[emotion_colm_name] = df[emotion_colm_name].replace("fear", fear_name)
    
    return df
