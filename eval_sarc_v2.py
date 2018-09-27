import os
import argparse
from glob import glob
import json
import csv
import numpy as np

from src import config


def get_ids_and_ndarray_from_prob_file(filename):
    probs_tuple_pairs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        probs_tuple_pair = []
        for i, str_probs in enumerate(lines):
            numeric_probs_tuple = [float(elem) for elem in str_probs]
            if i % 2 == 1:
                probs_tuple_pair.append(numeric_probs_tuple)
                assert len(probs_tuple_pair) == 2
                probs_tuple_pair = np.array(probs_tuple_pair)
                probs_tuple_pairs.append(probs_tuple_pair)
                probs_tuple_pair = []
            else:
                probs_tuple_pair.append(numeric_probs_tuple)

    return probs_tuple_pairs


parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('--model_hash', help='Path to corpus folder')

args = parser.parse_args()


experiment_path = os.path.join(config.RESULTS_PATH, args.model_hash + '*')
ext_experiment_path = glob(experiment_path)
assert len(ext_experiment_path) == 1, 'Try providing a longer model hash'
ext_experiment_path = ext_experiment_path[0]

hyperparams_path = os.path.join(ext_experiment_path, 'hyperparams.json')

with open(hyperparams_path) as f:
    hyperparams = json.load(f)

corpus_name =  hyperparams['corpus']

if 'sarc-v2' not in corpus_name:
    raise ValueError('Dataset of trained model is not sarc-v2 type')

corpus_labels_dict = config.label_dict[corpus_name]

label_pairs = []
with open(corpus_labels_dict['test']) as f:
    label_pair = []
    for i, line in enumerate(f.readlines()):
        label = int(line.strip())
        if i % 2 == 1:
            label_pair.append(label)
            assert len(label_pair) == 2
            label_pair = np.asarray(label_pair)
            label_pairs.append(label_pair)
            label_pair = []
        else:
            label_pair.append(label)

test_probs_path = os.path.join(ext_experiment_path, 'test_probs.csv')

if not os.path.exists(test_probs_path):
    raise OSError(('Test probabilities file does not exist, make sure you have '
                   'executed run.py with the test parameters --test'))

probs_pairs = get_ids_and_ndarray_from_prob_file(test_probs_path)

assert len(label_pairs) == len(probs_pairs)

ok = 0
for label_pair, prob_pair in zip(label_pairs, probs_pairs):
    ironic_index = label_pair.argmax()
    pred_ironic_incex = prob_pair[:, 1].argmax()
    if ironic_index == pred_ironic_incex:
        ok += 1

print(ok/len(probs_pairs))
