from tqdm import tqdm
from collections import defaultdict, Counter

import numpy as np

def Naive_Bayes(sentence, df):
    first = True
    for word in sentence:
        if word in df.index:
            if first == True:
                first = False
                log_prob_pos = df.loc[word]["log(P(w_i|+)smooth)"]
            else:
                log_prob_pos += df.loc[word]["log(P(w_i|+)smooth)"]

    first = True
    for word in sentence:
        if word in df.index:
            if first == True:
                first = False
                log_prob_neg = df.loc[word]["log(P(w_i|-)smooth)"]
            else:
                log_prob_neg += df.loc[word]["log(P(w_i|-)smooth)"]
        
    if log_prob_pos > log_prob_neg:
        return 'Positive'
    else:
        return 'Negative'
    
def get_unique_words(sentences):
    unique_words_set = set()

    for sentence in tqdm(sentences, desc='Processing Sentences'):
        for word in sentence:
            unique_words_set.add(word) 

    return sorted(unique_words_set)

def get_word_count(sentences, unique_words_set, freq, index):
    unique_words_set = set(unique_words_set)


    # Loop through happy sentences
    for sentence in tqdm(sentences):
        # Count tokens in the sentence
        counter = Counter(sentence)
        # Update frequencies for tokens in unique_words_set
        for token, count in counter.items():
            if token in unique_words_set:
                freq[token][index] += count
    return freq


def get_word_prob(main_df, unique_words_list, df_happy, df_other):
    sum_plus = main_df['count(w_i, +)'].sum()
    sum_min = main_df['count(w_i, -)'].sum()

    main_df['P(w_i|+)'] = main_df['count(w_i, +)'] / sum_plus
    main_df['P(w_i|-)'] = main_df['count(w_i, -)'] / sum_min

    main_df["P(w_i|+)smooth"] = (main_df["count(w_i, +)"] +1) / (sum_plus + len(unique_words_list))
    main_df["P(w_i|-)smooth"] = (main_df["count(w_i, -)"] +1) / (sum_min + len(unique_words_list))

    p_positive = (len(df_happy) / len(df_happy) + len(df_other))
    p_negative = (len(df_other) / len(df_happy) + len(df_other))


    main_df["log(P(w_i|+)smooth)"] = np.log(main_df["P(w_i|+)smooth"])
    main_df["log(P(w_i|-)smooth)"] = np.log(main_df["P(w_i|-)smooth"])

    return main_df, p_positive, p_negative

def naive_bayes(sentence, df):
    first = True
    for word in sentence:
        if word in df.index:
            if first == True:
                first = False
                prob_pos = df.loc[word]["P(w_i|+)"]
            else:
                prob_pos *= df.loc[word]["P(w_i|+)"]

    first = True
    for word in sentence:
        if word in df.index:
            if first == True:
                first = False
                prob_neg = df.loc[word]["P(w_i|-)"]
            else:
                prob_neg *= df.loc[word]["P(w_i|-)"]
    if prob_pos > prob_neg:
        return("Happy")
    else:
        return("Other")
    
def naive_bayes_smooth(sentence, df):
    first = True
    for word in sentence:
        if word in df.index:
            if first == True:
                first = False
                log_prob_pos = df.loc[word]["log(P(w_i|+)smooth)"]
            else:
                log_prob_pos += df.loc[word]["log(P(w_i|+)smooth)"]

    first = True
    for word in sentence:
        if word in df.index:
            if first == True:
                first = False
                log_prob_neg = df.loc[word]["log(P(w_i|-)smooth)"]
            else:
                log_prob_neg += df.loc[word]["log(P(w_i|-)smooth)"]
        
    if log_prob_pos > log_prob_neg:
        return("Happy")
    else:
         return("Other")
    

def drop_unknown_sentences(df, main_df):
    # Create a list to store indices of sentences to remove
    indices_to_remove = []

    for index, tweet_list in df["sentence"].items():
        tweet = ' '.join(tweet_list)
        remove = True
        words_in_tweet = tweet.split()
        for word in words_in_tweet:
            if word in main_df.index:
                remove = False
                break
        if remove:
            indices_to_remove.append(index)

    return(df.drop(indices_to_remove))
