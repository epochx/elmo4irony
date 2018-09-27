import argparse
import os
from collections import namedtuple

from split import split_list

Tweet = namedtuple('Tweet', ['string', 'label', 'index'])

def read_platek_dataset(path):

    normal_tweets = []
    with open(os.path.join(path, 'en-balanced', 'normal.tweets.tsv')) as f:
        for line in f.readlines():
            index, string = line.strip().split('\t')
            if string != 'Not Available':
                tweet = Tweet(string, '0', index)
                normal_tweets.append(tweet)

    sarcastic_tweets = []
    with open(os.path.join(path, 'en-balanced', 'sarcastic.tweets.tsv')) as f:
        for line in f.readlines():
            index, string = line.strip().split('\t')
            if string != 'Not Available':
                tweet = Tweet(string, '1', index)
                sarcastic_tweets.append(tweet)

    max_len = min([len(normal_tweets), len(sarcastic_tweets)])

    tweets = normal_tweets[:max_len] + sarcastic_tweets[:max_len]

    return tweets


parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('path', help='Path to corpus folder')
parser.add_argument('destination_path',
                    help='Directory where prepared files will be saved')

args = parser.parse_args()

tweets = read_platek_dataset(args.path)

train_tweets, valid_tweets, test_tweets = split_list(tweets,
                                                     shuffle=True,
                                                     train_ratio=0.7,
                                                     valid_ratio=0.1,
                                                     test_ratio=0.2)

output_path = os.path.join(args.destination_path, 'platek-sarcasm')

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(os.path.join(output_path, 'train.txt'), 'w') as f:
    for tweet in train_tweets:
        f.write('{0}\t{1}\t{2}\n'.format(tweet.index,
                                         tweet.label,
                                         tweet.string))

with open(os.path.join(output_path, 'dev.txt'), 'w') as f:
    for tweet in valid_tweets:
        f.write('{0}\t{1}\t{2}\n'.format(tweet.index,
                                         tweet.label,
                                         tweet.string))

with open(os.path.join(output_path, 'test.txt'), 'w') as f:
    for tweet in test_tweets:
        f.write('{0}\t{1}\t{2}\n'.format(tweet.index,
                                         tweet.label,
                                         tweet.string))
