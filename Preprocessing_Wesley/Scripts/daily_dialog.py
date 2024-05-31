import pandas as pd
from config import Config

emotion_colm_name = Config.emotion_colm_name
sentence_colmn_name = Config.sentence_colmn_name
keep_neutral = Config.keep_neutral
data_path = Config.datapath


def load_daily_dialog_data():
    """
    Load Daily Dialog data from text files.

    Returns:
        DataFrame: DataFrame containing dialogues and corresponding emotions.
    """
    main_path = data_path + "ijcnlp_dailydialog/ijcnlp_dailydialog/"

    # Read dialogues
    with open(main_path + "dialogues_text.txt", "r") as file:
        data = file.readlines()

    # Read emotions
    with open(main_path + "dialogues_emotion.txt", "r") as file:
        emotions = file.readlines()

    # Flatten dialogues
    flattened_data = [item for sublist in data for item in sublist.split('__eou__') if item.strip() != '']
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data, columns=[sentence_colmn_name])

    # Extract emotions
    j = 0
    for x in emotions:
        for i in x:
            if i.isnumeric():
                df.at[j, 'emotion'] = int(i)
                j += 1

    # Process emotions
    if keep_neutral:
        df[emotion_colm_name] = df[emotion_colm_name].astype(str)
        df[emotion_colm_name] = df[emotion_colm_name].replace('0.0', "Neutral")
    else:
        df = df[df[emotion_colm_name] != 0.0]
        df[emotion_colm_name] = df[emotion_colm_name].astype(str)
    df[emotion_colm_name] = df[emotion_colm_name].replace('1.0', "Mad")
    df[emotion_colm_name] = df[emotion_colm_name].replace('2.0', "Disgust")
    df[emotion_colm_name] = df[emotion_colm_name].replace('3.0', "Scared")
    df[emotion_colm_name] = df[emotion_colm_name].replace('4.0', "Happy")
    df[emotion_colm_name] = df[emotion_colm_name].replace('5.0', "Sad")
    df[emotion_colm_name] = df[emotion_colm_name].replace('6.0', "Surprised")

    return df
