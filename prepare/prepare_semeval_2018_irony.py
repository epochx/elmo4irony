import os
import argparse


parser = argparse.ArgumentParser(description='', add_help=False)

parser.add_argument('path', help='Path to corpus folder')
parser.add_argument('destination_path',
                    help='Directory where prepared files will be saved')

args = parser.parse_args()

train_lines = []
dev_lines = []

train_file_path = os.path.join(args.path, 'datasets', 'train',
                               'SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt')

with open(train_file_path) as f:
    # skip header
    line = f.readline()
    for i, line in enumerate(f.readlines()):
        if i <= 3066:
            train_lines.append(line)
        else:
            dev_lines.append(line)


output_path = os.path.join(args.destination_path, 'semeval-2018-irony')

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(os.path.join(output_path, 'train.txt'), 'w') as f:
    for line in train_lines:
        f.write(line)

with open(os.path.join(output_path, 'dev.txt'), 'w') as f:
    for line in dev_lines:
        f.write(line)

test_file_path = os.path.join(args.path, 'datasets', 'goldtest_TaskA',
                              'SemEval2018-T3_gold_test_taskA_emoji.txt')

with open(test_file_path) as f:
    # skip header
    line = f.readline()
    with open(os.path.join(output_path, 'test.txt'), 'w') as out_f:
        for line in f.readlines():
            out_f.write(line)
