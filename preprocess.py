import re
import os
import sys

import argparse
from src.config import (PREPARED_DATA_PATH, PREPROCESSED_DATA_PATH,
                        URL_TOKEN, USR_TOKEN, HASHTAG_TOKEN)
from twokenize import tokenizeRawTweetText

url_regex = re.compile('(?P<url>https?://[^\s]+)')
user_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)')
hashtag_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)')

IRONIC_HASHTAGS = ['#irony', '#ironic', '#sarcasm', '#sarcastic', '#not']


def filter_hashtags(tokens):
    tokens = [token for token in tokens if token.lower() not in IRONIC_HASHTAGS]
    return tokens


def open_prepared_file(filepath):
    strings = []
    labels = []
    with open(filepath) as f:
        for line in f.readlines():
            try:
                index, label, string = line.strip().split('\t')
                strings.append(string)
                labels.append(label)
            except ValueError as e:
                print(line)
                print(e)
    return strings, labels


def preprocess(strings, labels, min_len=None, max_len=None):

    strings = [re.sub(url_regex, URL_TOKEN, string)
               for string in strings]

    strings = [re.sub(user_regex, USR_TOKEN, string)
               for string in strings]

    tokenized_strings = [tokenizeRawTweetText(string)
                         for string in strings]

    filtered_tokenized_strings = []
    filtered_labels = []
    for tokenized_string, label in zip(tokenized_strings, labels):
        filtered_tokens = filter_hashtags(tokenized_string)
        if min_len and len(filtered_tokens) < min_len:
            continue
        if max_len and len(filtered_tokens) > max_len:
            filtered_tokens = filtered_tokens[:max_len]

        filtered_tokenized_strings.append(filtered_tokens)
        filtered_labels.append(label)

    return filtered_tokenized_strings, filtered_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    datasets = [item for item in os.listdir(PREPARED_DATA_PATH) if
                os.path.isdir(os.path.join(PREPARED_DATA_PATH, item))]

    parser.add_argument("--dataset", required=True,
                        choices=datasets,
                        help="Input file paths")

    parser.add_argument("--augment",
                        nargs='*', help="Input file paths to add to train")

    parser.add_argument("--name", type=str,
                        help="Name the preprocessed dataset, otherwise use original")

    parser.add_argument('--min_len', type=int, default=None,
                        help='Minimum length to add to dataset')

    parser.add_argument('--max_len', type=int, default=None,
                        help='Maximum length to add to dataset')

    args = parser.parse_args()

    sys.stdout.write(f"Preprocessing {args.dataset}...\n")

    dataset_path = os.path.join(PREPARED_DATA_PATH, args.dataset)
    if not os.path.exists(dataset_path):
        raise IOError("Prepared dataset does not exist. Did you create it already?")

    train_strings, train_labels = \
        open_prepared_file(os.path.join(dataset_path, 'train.txt'))

    if args.augment:
        for input_file_path in args.augment:
            strings_i, labels_i = open_prepared_file(input_file_path)
            train_strings += strings_i
            train_labels += labels_i

    train_examples, train_labels = preprocess(train_strings,
                                              train_labels,
                                              min_len=args.min_len,
                                              max_len=args.max_len)

    valid_strings, valid_labels = \
        open_prepared_file(os.path.join(dataset_path, 'dev.txt'))
    valid_examples, valid_strings = preprocess(valid_strings,
                                               valid_strings,
                                               min_len=args.min_len,
                                               max_len=args.max_len)

    test_strings, test_labels = \
        open_prepared_file(os.path.join(dataset_path, 'test.txt'))

    test_examples, test_labels = preprocess(test_strings,
                                            test_labels,
                                            min_len=args.min_len,
                                            max_len=args.max_len)

    name = args.name if args.name else args.dataset

    if args.min_len:
        name = '{0}_min_{1}'.format(name, args.min_len)

    if args.max_len:
        name = '{0}_max_{1}'.format(name, args.max_len)

    output_path = os.path.join(PREPROCESSED_DATA_PATH, name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'train.txt'), 'w') as f:
        for tokens in train_examples:
            f.write('{}\n'.format(' '.join(tokens)))

    with open(os.path.join(output_path, 'train_labels.txt'), 'w') as f:
        for label in train_labels:
            f.write('{}\n'.format(label))


    with open(os.path.join(output_path, 'dev.txt'), 'w') as f:
        for tokens in valid_examples:
            f.write('{}\n'.format(' '.join(tokens)))

    with open(os.path.join(output_path, 'dev_labels.txt'), 'w') as f:
        for label in valid_labels:
            f.write('{}\n'.format(label))


    with open(os.path.join(output_path, 'test.txt'), 'w') as f:
        for tokens in test_examples:
            f.write('{}\n'.format(' '.join(tokens)))

    with open(os.path.join(output_path, 'test_labels.txt'), 'w') as f:
        for label in test_labels:
            f.write('{}\n'.format(label))
