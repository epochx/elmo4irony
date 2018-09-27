#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import argparse
import tqdm
import json

import twitter


def clean_tweet(text):
    return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Twiter API tweet crawler")

    parser.add_argument('--input',
                        required=True)

    TWITTER_CREDS = '.twitter_credentials.conf'

    if not os.path.exists(TWITTER_CREDS):
        raise OSError('Credentials File Not Found')

    with open(TWITTER_CREDS) as f:
        credentials = json.load(f)

    for key, value in credentials.items():
        if len(value) == 0:
            raise Exception(f'Missing {key} in {TWITTER_CREDS}')

    api = twitter.Api(**credentials,
                      sleep_on_rate_limit=True)

    args = parser.parse_args()

    input_file_path = args.input

    assert args.input.endswith('.txt')

    output_file_path = args.input.replace('.txt', '.tweets.tsv')

    with open(input_file_path) as in_file:
        tids = [line.strip() for line in in_file.readlines()]

    cached_tids = []
    if os.path.exists(output_file_path):
        with open(output_file_path) as out_file:
            cached_tids = [line.strip().split('\t')[0]
                           for line in out_file.readlines()]

    with open(output_file_path, 'a') as out_file:
        for tid in tqdm.tqdm(tids):
            if cached_tids and tid in cached_tids:
                continue

    with open(output_file_path, 'a') as out_file:

        for tid in tqdm.tqdm(tids):

            if cached_tids and tid in cached_tids:
                continue

            try:
                tweet = api.GetStatus(tid)
                text = clean_tweet(tweet.text)

            except twitter.error.TwitterError as e:
                text = 'Not Available'

            out_file.write('{0}\t{1}\n'.format(tid, text))
            out_file.flush()
