# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:28:20 2022

@author: hp
"""

import csv
import re

import twitter
import twitter_credentials

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# initialize api instance
twitter_api=twitter.Api(consumer_key=twitter_credentials.CONSUMER_KEY,
                        consumer_secret=twitter_credentials.CONSUMER_SECRET,
                        access_token_key=twitter_credentials.ACCESS_TOKEN,
                        access_token_secret=twitter_credentials.ACCESS_TOKEN_SECRET)


# test authentication
# print(twitter_api.VerifyCredentials())

def build_test_set(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count=100)

        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)

        return [{"text": status.text, "label": None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None


search_term = input("Enter a search keyword: ")
test_data_set = build_test_set(search_term)

training_data_set = []
with open("tweetDataFile.csv", 'rt') as csvfile:
    lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
    for row in lineReader:
        training_data_set.append({"tweet_id": row[0], "text": row[1], "label": row[2], "topic": row[3]})


class PreprocessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def process_tweets(self, list_of_tweets):
        processed_tweets = []
        for tweet in list_of_tweets:
            if tweet["label"] is not None:
                if tweet["label"] == "positive" or tweet["label"] == "negative":
                    processed_tweets.append((self._process_tweet(tweet["text"]), tweet["label"]))
            else:
                processed_tweets.append((self._process_tweet(tweet["text"]), None))

        return processed_tweets

    def _process_tweet(self, tweet):
        tweet = tweet.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)

        words = []
        for word in tweet:
            if word not in self._stopwords:
                words.append(word)
        return words


tweet_processor = PreprocessTweets()
preprocessed_training_set = tweet_processor.process_tweets(training_data_set)
preprocessed_test_set = tweet_processor.process_tweets(test_data_set)


def build_vocabulary(preprocessed_training_data):
    all_words = []

    for (words, sentiment) in preprocessed_training_data:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features


training_data_features = build_vocabulary(preprocessed_training_set)


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in training_data_features:
        is_feature_in_words = word in tweet_words
        features[word] = is_feature_in_words
    return features


training_features = nltk.classify.apply_features(extract_features, preprocessed_training_set)

NBayesClassifier = nltk.NaiveBayesClassifier.train(training_features)

# label = NBayesClassifier.classify(extract_features("I am happy"))

classified_result_labels = []
for tweet in preprocessed_test_set:
    classified_result_labels.append(NBayesClassifier.classify(extract_features(tweet[0])))

if classified_result_labels.count('positive') > classified_result_labels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(
        100 * classified_result_labels.count('positive') / len(classified_result_labels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(
        100 * classified_result_labels.count('negative') / len(classified_result_labels)) + "%")