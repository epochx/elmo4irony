#!/usr/bin/env python3

import random
import argparse
random.seed(42)


def split_list(examples, shuffle=False, train_ratio=0.7, valid_ratio=0.3, test_ratio=0.0):
    """
    :param examples: list
    :param train_ratio: 0.8
    :return:
    """
    if test_ratio:
        assert 1.0 - 1e-8 <= train_ratio + valid_ratio + test_ratio <= 1.0 + 1e-8, "ratios don't sum to 1"
    else:
        assert train_ratio + valid_ratio == 1.0, "ratios don't sum 1"

    num_examples = len(examples)

    if shuffle:
        random.shuffle(examples)

    last_train_idx = int(num_examples * train_ratio)
    last_valid_idx = int(last_train_idx + (num_examples * valid_ratio))

    train_data = examples[:last_train_idx]
    if test_ratio:
        valid_data = examples[last_train_idx:last_valid_idx]
        test_data = examples[last_valid_idx:]
        return train_data, valid_data, test_data
    else:
        valid_data = examples[last_train_idx:]
        return train_data, valid_data, valid_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", default=None, required=True,
                        type=str, help="Input file path")

    parser.add_argument("--train_file", "-trf", default=None, required=True,
                        type=str, help="Train file path")
    parser.add_argument("--valid_file", "-vaf", default=None, required=True,
                        type=str, help="Train file path")
    parser.add_argument("--test_file", "-tef", default=None,
                        type=str, help="Test file path")

    parser.add_argument("--train_ratio", "-trr", default=0.7, required=True,
                        type=float, help="Train ratio")
    parser.add_argument("--valid_ratio", "-var", default=0.3, required=True,
                        type=float, help="Valid ratio")
    parser.add_argument("--test_ratio", "-ter", default=0.0,
                        type=float, help="Test ratio")

    parser.add_argument("--shuffle", "-s", action="store_true", default=False,
                        help="Whether to shuffle input or not")

    pargs = parser.parse_args()

    if pargs.test_ratio != 0.0 and pargs.test_file is None:
        raise RuntimeError("non-zero --test_ratio but no --test_file provided")

    if pargs.test_ratio == 0.0 and pargs.test_file is not None:
        raise RuntimeError("--test_file provided but --test_ratio equals zero")

    with open(pargs.input_file, 'r') as f:
        lines = f.readlines()

    train, valid, test = split_list(
        lines,
        train_ratio=pargs.train_ratio,
        valid_ratio=pargs.valid_ratio,
        test_ratio=pargs.test_ratio,
        shuffle=pargs.shuffle,
    )

    with open(pargs.train_file, 'w') as f:
        f.writelines(train)

    with open(pargs.valid_file, 'w') as f:
        f.writelines(valid)

    if pargs.test_ratio != 0.0:
        with open(pargs.test_file, 'w') as f:
            f.writelines(test)
