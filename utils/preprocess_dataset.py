import pandas as pd
import numpy as np
import pandas as pd
from collections import Counter
import datetime 
from nltk.corpus import stopwords

def most_common_words(texts, n_words=10):
    all_words = ' '.join(texts)
    stop_words = set(stopwords.words('english'))
    words = [w for w in all_words.split() if len(w) > 1 and w not in stop_words]
    return Counter(words).most_common(n_words)

def most_unique_common_words(texts, words_to_exclude,n_words=5):
    all_words = ' '.join(texts)
    stop_words = set(stopwords.words('english'))
    for element in words_to_exclude:
        stop_words.add(element[0])
    words = [w for w in all_words.split() if len(w) > 1 and w not in stop_words]
    return Counter(words).most_common()[:5]

def preprocess_dataframe(input_dataset, mode='train'):
    input_df = pd.read_csv(input_dataset)

    input_df["tweet_length"] = input_df["Tweet"].apply(len)
    input_df["tweet_n_words"] = input_df["Tweet"].apply(lambda x: len(x.split()))

    total_dataset_most_common_words = most_common_words(input_df["Tweet"], 50)

    if mode == 'train':
        output_df = input_df.groupby('ID').agg({'Timestamp': list,'Tweet': list, 'EventType': list, "tweet_length": np.mean, "tweet_n_words": np.mean}).reset_index()
        output_df["EventType"] = output_df["EventType"].apply(lambda x: x[0])
    else:
        output_df = input_df.groupby('ID').agg({'Timestamp': list,'Tweet': list, "tweet_length": np.mean, "tweet_n_words": np.mean}).reset_index()

    output_df["n_tweets"] = output_df["Tweet"].apply(lambda x: len(x))
    output_df["tweet_length"] = output_df["tweet_length"].apply(lambda x: round(x, 2))
    output_df["tweet_n_words"] = output_df["tweet_n_words"].apply(lambda x: round(x, 2)) 

    output_df = output_df.sort_values("ID", key=(lambda x: x.apply(lambda y: int(y.split('_')[1]))))
    output_df.reset_index(drop=True, inplace=True)

    output_df["Timestamp"] = output_df["Timestamp"].apply(lambda x: [datetime.datetime.fromtimestamp(y/1000) for y in x])
    output_df["Unique Common Words"] = output_df["Tweet"].apply(lambda x: most_unique_common_words(x, total_dataset_most_common_words))

    return output_df

