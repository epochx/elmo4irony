import argparse
import os
from collections import namedtuple

from split import split_list

Tweet = namedtuple('Tweet', ['string', 'label', 'index'])


def read_iac_v1_dataset(path):
    v1_data = []

    sarc_path = os.path.join(path, 'sarc')
    for file_name in os.listdir(sarc_path):
        if file_name != '.DS_Store':
            file_path = os.path.join(sarc_path, file_name)
            with open(file_path, encoding='latin1') as f:
                text = f.read().strip().replace('\t', ' ')
                tweet = Tweet(text, '1', file_name.replace('.txt', ''))
                v1_data.append(tweet)

    notsarc_path = os.path.join(path, 'notsarc')
    for file_name in os.listdir(notsarc_path):
        if file_name != '.DS_Store':
            file_path = os.path.join(notsarc_path, file_name)
            with open(file_path) as f:
                text = f.read().strip().replace('\t', ' ')
                tweet = Tweet(text, '0', file_name.replace('.txt', ''))
                v1_data.append(tweet)

    return v1_data


parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('path', help='Path to corpus folder')
parser.add_argument('destination_path',
                    help='Directory where prepared files will be saved')

args = parser.parse_args()

tweets = read_iac_v1_dataset(args.path)

train_tweets, valid_tweets, test_tweets = split_list(tweets,
                                                     shuffle=True,
                                                     train_ratio=0.7,
                                                     valid_ratio=0.1,
                                                     test_ratio=0.2)

output_path = os.path.join(args.destination_path, 'iac-v1')

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
