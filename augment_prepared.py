# coding: utf-8

import re
import os
import argparse

from guess_language import guess_language
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from collections import namedtuple

from src.config import PREPARED_DATA_PATH


url_regex = re.compile('(?P<url>https?://[^\s]+)')
user_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)')
hashtag_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)')

Tweet = namedtuple('Tweet', ['string', 'label', 'index'])


def open_prepared_file(filepath):
    tweets = []
    with open(filepath) as f:
        for line in f.readlines():
            index, label, string = line.strip().split('\t')
            tweet = Tweet(string, label, index)
            tweets.append(tweet)
    return tweets


def open_extra_file(filepath, ironic=False):
    tweets = []
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            string = line.strip()
            label = '1' if ironic else '0'
            index = '{0}@{1}'.format(i, filepath)
            tweet = Tweet(string, label, index)
            tweets.append(tweet)
    return tweets


def open_semeval_2018_file(filepath):
    tweets = []
    strings = set()
    with open(filepath) as f:
        for line in f.readlines():
            index, string = line.strip().split('\t')
            if string in strings:
                continue
            strings.add(string)
            tweet = Tweet(string, '0', index)
            tweets.append(tweet)
    return tweets


def is_english(tweet):
    lang_1 = guess_language(tweet.string)
    try:
        lang_2 = detect(tweet.string)
    except LangDetectException:
        lang_2 = 'UNKNOWN'

    if lang_1 == 'en' and lang_2 == 'en':
        return True

    return False


def extract_hashtags_list(tweets):
    hashtags = []
    for tweet in tweets:
        hashtags += re.findall(hashtag_regex, tweet.string)
    return hashtags


def extract_users_list(tweets):
    users = []
    for tweet in tweets:
        users += re.findall(user_regex, tweet.string)
    return users



parser = argparse.ArgumentParser(description='', add_help=False)


datasets = [item for item in os.listdir(PREPARED_DATA_PATH) if
                os.path.isdir(os.path.join(PREPARED_DATA_PATH, item))]

parser.add_argument("--dataset", required=True,
                    choices=datasets,
                    help="Input dataset")

parser.add_argument("--ironic", required=True,
                    nargs='*', help="Input file paths")

parser.add_argument("--nonironic", required=True,
                    nargs='*', help="Input file paths")

parser.add_argument("--name", required=True,
                    type=str, help="Name of the augmented dataset")

args = parser.parse_args()

dataset_tweets = open_prepared_file(os.path.join(PREPARED_DATA_PATH,
                                                 args.dataset,
                                                 'train.txt'))

ironic_tweets = []
for ironic_file_path in args.ironic:
    ironic_tweets += open_extra_file(ironic_file_path, ironic=True)

nonironic_tweets = []
for nonironic_file_path in args.nonironic:
    nonironic_tweets += open_extra_file(nonironic_file_path)

print('')
print('Filtering out tweets with newlines on them...')
new_line_filter = lambda x: not '\n' in x.string
ironic_tweets = list(filter(new_line_filter, ironic_tweets))
nonironic_tweets = list(filter(new_line_filter, nonironic_tweets))

print('Filtering out tweets with different hashtags...')
dataset_hashtags = extract_hashtags_list(dataset_tweets)
dataset_hashtags = {'#{0}'.format(item) for item in dataset_hashtags
                    if item.lower() not in ['irony', 'sarcasm']}
hashtag_filter = lambda x: any([hashtag in x.string
                                for hashtag in dataset_hashtags])
ironic_tweets = list(filter(hashtag_filter, ironic_tweets))
nonironic_tweets = list(filter(hashtag_filter, nonironic_tweets))

print('Filtering out non English tweets..')
ironic_tweets = list(filter(is_english, ironic_tweets))
nonironic_tweets = list(filter(is_english, nonironic_tweets))

max_len = min([len(ironic_tweets), len(nonironic_tweets)])
print('Balancing size to {}...'.format(max_len))

ironic_tweets = ironic_tweets[:max_len]
nonironic_tweets = nonironic_tweets[:max_len]

output_ironic_file_path = '{0}_{1}'.format(args.name, 'ironic.txt')
output_nonironic_file_path = '{0}_{1}'.format(args.name, 'nonironic.txt')

with open(output_ironic_file_path, 'w') as f:
    for tweet in ironic_tweets:
        f.write('{0}\t{1}\t{2}\n'.format(tweet.index,
                                         tweet.label,
                                         tweet.string))


with open(output_nonironic_file_path, 'w') as f:
    for tweet in nonironic_tweets:
        f.write('{0}\t{1}\t{2}\n'.format(tweet.index,
                                         tweet.label,
                                         tweet.string))

print('Done')
print('')
