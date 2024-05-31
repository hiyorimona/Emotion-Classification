import re
import os
import string
import pandas as pd
from tqdm import tqdm
from zipfile import BadZipFile
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import keras
import spacy


tqdm.pandas()
# load spacy
nlp = spacy.load('en_core_web_sm')


def data_reader(data_path):
    """
    Function to read CSV files from directories and define DataFrames
    with variable names based on directory names.

    :param data_path: Path to the directory containing CSV files.
    :return: List of DataFrames <dfs> and corresponding keys <keys>.
    """
    folders = os.listdir(data_path)
    csv_files = {}
    for folder in folders:
        try:
            folder_path = os.path.join(data_path, folder)
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(folder_path, file)
                    csv_files[folder] = csv_path
                    break

        except NotADirectoryError:
            print(f'skipping {folder}!')

    dfs = []
    keys = []

    for key, path in csv_files.items():
        if key.endswith('.csv'):
            key = key[:-4]
            try:
                df = pd.read_csv(path, compression='zip')
                dfs.append(df)
                keys.append(key)
                print(f'{key} loaded!')
            except BadZipFile:
                df = pd.read_csv(path, sep='\t')
                dfs.append(df)
                keys.append(key)
                print(f'{key} loaded!')
        else:
            try:
                df = pd.read_csv(path, compression='zip')
                dfs.append(df)
                keys.append(key)
                print(f'{key} loaded!')
            except BadZipFile:
                df = pd.read_csv(path)
                dfs.append(df)
                keys.append(key)
                print(f'{key} loaded!')



    return dfs,keys

def clean_string(text, stem="None"):
    """
    Clean the input text by performing various preprocessing steps.

    :param text: The input text from the DataFrame to be cleaned.
    :param stem: The stemming or lemmatization method to be applied. Options: 'None', 'Stem', 'Lem', 'Spacy'.
    Default is 'None'.
    :return: The cleaned text <final_string> after applying preprocessing steps.
    """

    final_string = ""

    # lowercase
    text = text.lower()

    # regex pattern to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    # Remove emojis from the text
    text_without_emojis = emoji_pattern.sub(r'', text)
    # Remove curly quotes
    text_without_quotes = text_without_emojis.replace('“', '').replace('”', '')
    # Remove emphasis
    text_without_emphasis = text_without_quotes.replace('’', '').replace('‘', '')
    # Remove dashes
    text_without_dashes = text_without_emphasis.replace("''——", "").replace("''—", "")


    # Remove line breaks
    # Note: that this line can be augmented and used over
    # to replace any characters with nothing or a space
    text = re.sub(r'\n', '', text_without_dashes)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]


    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string

def data_prep(df,stem="None"):
    """
    Perform data preprocessing on the DataFrame.

    :param df: DataFrame to be preprocessed.
    :return: Preprocessed DataFrame.
    """
    
    df['sentence'] = df['sentence'].progress_apply(lambda x: clean_string(x,stem="None"))
    df = df[df['sentence'] != '']
    df = df.drop_duplicates(subset=['sentence'], keep=False).reset_index(drop=True)
    df.replace({'emotion': {'scared': 'fear'}}, inplace=True)

    print(df['emotion'].value_counts())
    df = df.sample(frac=1)
    
    return df

