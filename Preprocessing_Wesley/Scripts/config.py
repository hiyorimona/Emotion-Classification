class Config(object):
    datapath = "../folders/"

    ##################
    # Merging of data#
    ##################

    emotion_colm_name = "emotion"
    sentence_colmn_name = "sentence"


    keep_neutral = False

    ###  emotion names

    sad = "sadness"
    happy = "happiness"
    mad = "anger"
    surpised = "surprise"
    disgusted = "disgust"
    scared = "scared"

    ### Fairy tails

    suprised_positive = "Surprised"
    suprised_negative = "Surprised"



    ### GoEmotions

    priotize_happy = False

    ####################
    ### Keep datasets###
    ####################
    #Specify which datasets you want to add to the final dataset.

    affectivetext = False
    carer = False
    chatgpt_generated = True
    daily_dialog = False
    emotion_detection_master = False
    fairy_tails = False
    goeomotions = False
    meld_masters = False
    survivor = False



    ##################
    # Preprocessing  #
    ##################

    ### Binary

    happiness_other = False

    evenly_distributed = False

    remove_stopwords = False
    stemmer = False

    ### Sentiment Extraction
    extract_sentiment = False
    
    sentiment_colmn_name = "sentiment"


    sentiment_library = "nltk" # spacyy, nltk

    ### NLTK
    # Only one of these can be True
    keep_only_compound = True # Keep only the compount score.
    get_positive = False # if True, it will return Postive or Negative based on the positive and negative score.
    get_highest = False # If true it will return the highest: Positive, Negative or Neutral

    # If all are false, all of the scores will be returned.


    ### Feature extraction
    NER_extraction = False
    NER_colmn_name = "NER_tag"

    POS_extraction = False
    POS_colmn_name = "POS_tag"