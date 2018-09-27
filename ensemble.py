#!/usr/bin/python

import argparse
import csv
import numpy as np
import pandas as pd

from src.config import LABEL2ID

parser = argparse.ArgumentParser(
                   description='Obtain predicted classes by averaging '
                               'several softmax outputs.')

parser.add_argument('files', type=str, nargs='+',
                    help='prob files from which to generate the predictions')

parser.add_argument('--output', type=str,
                    default='ensembled_predictions.txt',
                    help='Name of output file')

parser.add_argument('--mode', default='mean',
                    choices=['mean', 'hard-voting', 'soft-voting'])


def get_ids_and_ndarray_from_prob_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        numeric_probs = [[float(elem) for elem in str_probs] for str_probs in lines]
    return np.array(numeric_probs)


def run_main():
    args = parser.parse_args()
    print(args)
    filenames = args.files
    array_list = []

    for filename in filenames:
        probs_array = get_ids_and_ndarray_from_prob_file(filename)
        array_list.append(probs_array)

    tensor = np.stack(array_list, axis=2)

    if args.mode == 'mean':
        mean_probs = np.mean(tensor, axis=2)
        label_ids = np.argmax(mean_probs, axis=1).tolist()

    elif args.mode == 'hard-voting':
        # this assumes that we are feeding IDs labels
        votes = tensor.argmax(1)
        pd_votes = pd.DataFrame(votes)
        pd_label_ids = pd_votes.apply(lambda x: x.value_counts(), axis=1).fillna(0)
        label_ids = pd_label_ids.values.argmax(1)

    elif args.mode == 'soft-voting':
        sum_probs = tensor.sum(2)
        label_ids = sum_probs.argmax(1)


    id2label = {v: k for k, v in LABEL2ID.items()}
    labels = [id2label[label_id] for label_id in label_ids]

    preds_filename = args.output
    print('Writing {}'.format(preds_filename))
    with open(preds_filename, 'w') as f:
        for pred_label in labels:
            f.write(pred_label + '\n')


if __name__ == "__main__":
    run_main()
