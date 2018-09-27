import argparse
import csv
import os
from collections import namedtuple

from split import split_list

Tweet = namedtuple('Tweet', ['string', 'label', 'index'])

def read_iac_v2_dataset(path):
    #rows have keys: 'Corpus', 'Label', 'ID', 'Quote Text', 'Response Text'
    v2_data = []
    with open(path) as f:
        rows = csv.DictReader(f)
        for row in rows:
            tweet = Tweet(row['Response Text'],
                          '0' if row['Label'] == 'notsarc' else '1',
                          row['ID'])
            v2_data.append(tweet)

    return v2_data

parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('path', help='Path to corpus folder')

args = parser.parse_args()

tweets = read_iac_v2_dataset(args.path)

train_tweets, valid_tweets, test_tweets = split_list(tweets,
                                                     shuffle=True,
                                                     train_ratio=0.7,
                                                     valid_ratio=0.1,
                                                     test_ratio=0.2)

output_path = 'iac-v2'


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
