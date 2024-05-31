from nltk.corpus import stopwords          
from nltk.stem import PorterStemmer        
from nltk.tokenize import TweetTokenizer
import nltk
import string
import re

nltk.download('stopwords')
stopwords_english = stopwords.words('english')
stemer = PorterStemmer()
def Tokenizer(text, remove_stopwords = False, stemmer= False):
    processed_text =  re.sub('http\\S+', '', text)
    processed_text = re.sub('#', '', processed_text)

    tokenize = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    processed_text = tokenize.tokenize(processed_text)
    if remove_stopwords:
        processed_text = [word for word in processed_text if word not in stopwords_english and word not in string.punctuation]

    if stemmer:
    
        processed_text = [stemer.stem(word) for word in processed_text]


    return processed_text
