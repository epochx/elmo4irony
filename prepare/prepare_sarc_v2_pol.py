import argparse
import csv
import json
import os
from collections import namedtuple

from split import split_list

Tweet = namedtuple('Tweet', ['string', 'label', 'index'])

parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('path', help='Path to corpus folder')
parser.add_argument('destination_path',
                    help='Directory where prepared files will be saved')

args = parser.parse_args()


def read_csv_file(file_path, json_data):
    tweets = []
    with open(file_path) as f:
        for row in csv.reader(f):
            comment_ids, response_ids, labels = row[0].split('|')
            response_ids = response_ids.split()
            labels = labels.split()
            for response_id, label in zip(response_ids, labels):
                tweet = Tweet(json_data[response_id]['text'], label, response_id)
                tweets.append(tweet)
    return tweets


json_file_path = os.path.join(args.path, 'pol', 'comments.json')
train_csv_file_path = os.path.join(args.path, 'pol', 'train-balanced.csv')
test_csv_file_path = os.path.join(args.path, 'pol', 'test-balanced.csv')

with open(json_file_path) as f:
    json_data = json.load(f)


dev_tweets = read_csv_file(train_csv_file_path, json_data)
test_tweets = read_csv_file(test_csv_file_path, json_data)

train_tweets, valid_tweets, _ = split_list(dev_tweets,
                                           shuffle=True,
                                           train_ratio=0.8,
                                           valid_ratio=0.2,
                                           test_ratio=0.0)

output_path = os.path.join(args.destination_path, 'sarc-v2-pol')

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
