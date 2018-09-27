import argparse
import os
from collections import namedtuple

from split import split_list

Tweet = namedtuple('Tweet', ['string', 'label', 'index'])


def read_riloff_dataset(path):
    tweets_file_path = os.path.join(path, 'sarcasm-annos-emnlp13.tweets.tsv')
    labels_file_path = os.path.join(path, 'sarcasm-data', 'sarcasm-annos-emnlp13.tsv')

    tweets = []
    strings = []

    with open(tweets_file_path) as f:
        for line in f.readlines():
            index, string = line.strip().split('\t')
            strings.append((index, string))

    labels = []
    with open(labels_file_path) as f:
        for line in f.readlines():
            label = '0' if line.strip() == 'NOT_SARCASM' else '1'
            labels.append(label)

    assert len(strings) == len(labels)

    for (index, string), label in zip(strings, labels):
        if string != 'Not Available':
            tweet = Tweet(string, label, index)
            tweets.append(tweet)

    return tweets


parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('path', help='Path to corpus folder')
parser.add_argument('destination_path',
                    help='Directory where prepared files will be saved')

args = parser.parse_args()

tweets = read_riloff_dataset(args.path)

train_tweets, valid_tweets, test_tweets = \
    split_list(tweets, shuffle=True,
               train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2)

output_path = os.path.join(args.destination_path, 'riloff-sarcasm-data')

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
