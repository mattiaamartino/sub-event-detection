import pandas as pd
import numpy as np
import pandas as pd
from collections import Counter
import datetime 
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")
nltk.download('stopwords')


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

def add_top_words_column(df, tweet_col="Tweet", id_col="ID", max_df = 0.80, top_n=50):
    """
    Transforms the dataframe by adding a column with the top N words based on TF-IDF scores.
    """
    # Ensure the tweet column is joined into a single string per ID
    df["CombinedTweets"] = df[tweet_col].apply(lambda tweets: " ".join(tweets))
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=max_df, stop_words='english')
    
    # Compute the TF-IDF matrix for the entire dataset
    tfidf_matrix = vectorizer.fit_transform(df["CombinedTweets"])
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Create a new column for the top N words
    def extract_top_words(idx):
        # Get the TF-IDF scores for the current period
        scores = tfidf_matrix[idx].toarray().flatten()
        
        # Pair scores with feature names
        word_scores = list(zip(feature_names, scores))
        
        # Sort words by their TF-IDF scores in descending order
        ranked_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
        
        # Return top N words and scores separately
        top_words = [word for word, _ in ranked_words[:top_n]]
        top_scores = [float(score) for _, score in ranked_words[:top_n]]
        return top_words, top_scores

    # Apply the function to each period
    words_and_scores = [extract_top_words(idx) for idx in range(len(df))]
    df["TopWords"] = [w for w, _ in words_and_scores]
    df["TopWordScores"] = [np.array(s)/np.sum(s) for _, s in words_and_scores]

    # Drop the combined tweets column (optional)
    df.drop(columns=["CombinedTweets"], inplace=True)
    
    return df

def pre_processing_feature_extraction(input_dataset, mode='train', max_df = 0.8, top_n_words=200):
    """
    Preprocesses the dataframe, adding top TF-IDF-ranked words for each period.
    
    Parameters:
    input_dataset (str): Path to the input dataset (CSV file).
    mode (str): Whether the data is in 'train' or 'test' mode.
    top_n_words (int): Number of top words to extract using TF-IDF.
    
    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    # Load the dataset
    input_df = pd.read_csv(input_dataset)

    # Add tweet length and word count metrics
    input_df["tweet_length"] = input_df["Tweet"].apply(len)
    input_df["tweet_n_words"] = input_df["Tweet"].apply(lambda x: len(x.split()))

    # Group by 'ID' to consolidate tweets within the same period
    if mode == 'train':
        output_df = input_df.groupby('ID').agg({'Timestamp': list, 
                                                'Tweet': list, 
                                                'EventType': list, 
                                                "tweet_length": np.mean, 
                                                "tweet_n_words": np.mean}).reset_index()
        output_df["EventType"] = output_df["EventType"].apply(lambda x: x[0])
    else:
        output_df = input_df.groupby('ID').agg({'Timestamp': list, 
                                                'Tweet': list, 
                                                "tweet_length": np.mean, 
                                                "tweet_n_words": np.mean}).reset_index()

    # Add additional metrics
    output_df["n_tweets"] = output_df["Tweet"].apply(lambda x: len(x))
    output_df["tweet_length"] = output_df["tweet_length"].apply(lambda x: round(x, 2))
    output_df["tweet_n_words"] = output_df["tweet_n_words"].apply(lambda x: round(x, 2)) 

    # Sort the dataframe by numerical order of IDs
    output_df = output_df.sort_values("ID", key=(lambda x: x.apply(lambda y: int(y.split('_')[1]))))
    output_df.reset_index(drop=True, inplace=True)

    # Convert Timestamps from UNIX to datetime
    output_df["Timestamp"] = output_df["Timestamp"].apply(lambda x: [datetime.datetime.fromtimestamp(y/1000) for y in x])

    # Add top TF-IDF-ranked words
    output_df = add_top_words_column(output_df, tweet_col="Tweet", id_col="ID", max_df=max_df, top_n=top_n_words)

    return output_df