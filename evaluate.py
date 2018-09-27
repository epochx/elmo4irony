
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src import config

parser = argparse.ArgumentParser()

parser.add_argument('--corpus', type=str,
                    help='prob files from which to generate the predictions')

parser.add_argument('--predictions', type=str,
                    help='Name of output file')

parser.add_argument('--all', action='store_true')

args = parser.parse_args()



corpus_name =  args.corpus

corpus_labels_dict = config.label_dict[corpus_name]

true_labels = []
with open(corpus_labels_dict['test']) as f:
    for line in f.readlines():
        if line != '':
            label = int(line.strip())
            true_labels.append(label)

pred_labels = []
with open(args.predictions) as f:
    for line in f.readlines():
        if line != '':
            prediction = int(line.strip())
            pred_labels.append(prediction)


pred_labels = np.array(pred_labels)
true_labels = np.array(true_labels)

p, r, f1, s = precision_recall_fscore_support(true_labels,
                                              pred_labels,
                                              labels=[0, 1],
                                              average='macro')


print(f1)

if args.all:
    print(p)
    print(r)

    a = accuracy_score(true_labels, pred_labels)

    print(a)