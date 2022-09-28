# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 19:08:17 2022

@author: hp
"""

import twitter_credentials
import csv
import twitter
from twitter import TwitterError #Exception handling

twitter_api=twitter.Api(consumer_key=twitter_credentials.CONSUMER_KEY,
                        consumer_secret=twitter_credentials.CONSUMER_SECRET,
                        access_token_key=twitter_credentials.ACCESS_TOKEN,
                        access_token_secret=twitter_credentials.ACCESS_TOKEN_SECRET)
corpusFile="corpus.csv"
tweetDataFile="tweetDataFile.csv"
corpus=[]
with open(corpusFile,'rt')as csvfile:
    lineReader=csv.reader(csvfile,delimiter=',',quotechar="\"")
    for row in lineReader:
        corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})
        
trainingDataSet=[]
for tweet in corpus:
    try:
        #Get the content of tweet by ID
        status=twitter_api.GetStatus(tweet["tweet_id"])
        print("tweet fetched: "+status.text)
        tweet["text"]=status.text
        trainingDataSet.append(tweet)
    except TwitterError as err:
        print("Error while fetching tweet id "+tweet['tweet_id']+ ". Message: "+err.message[0]['message'])
        
# Now we write them to the empty CSV file
with open(tweetDataFile, 'wt') as csvfile:
    linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
    for tweet in trainingDataSet:
        try:
            linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
        except Exception as e:
            print(e)