
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords   
from spacy.lang.en import English
import spacy
from nltk.stem import PorterStemmer
from nltk import pos_tag



stemmer = PorterStemmer()
nltk.download('stopwords')
nlp = spacy.load("en_core_web_lg")
nlp = English()
sia = SentimentIntensityAnalyzer()


def remove_stopwords(sentence): 
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_sentence = [word for word in words if word.lower() not in stop_words]

    return ' '.join(filtered_sentence)



def get_max_sentiment(sentiment_dict):
    return max(sentiment_dict, key=sentiment_dict.get)

def extract_sentiment_nltk(sentence, keep_only_compound, get_positive, get_highest):
    if keep_only_compound:
        return sia.polarity_scores(sentence)["compound"] > 0
    elif get_positive:
        score = sia.polarity_scores(sentence)
        if score["pos"] > score["neg"]:
            return "Positive"
        else:
            return "Negative"
    elif get_highest:
        score = sia.polarity_scores(sentence)
        return get_max_sentiment(score)
    else:
        return sia.polarity_scores(sentence)

def apply_stemming(sentence):
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def nltk_pos_tag(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_tags = [tag for _, tag in tagged_tokens]
    return pos_tags

def spacy_ner_tag(text):
    doc = nlp(text)
    ner_tags = [(ent.text, ent.label_) for ent in doc.ents]
    return ner_tags
