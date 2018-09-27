#!/usr/bin/env bash

PREPROCESSED_DIR="$HOME/data/elmo4irony/preprocessed"

if [ ! -d "$PREPROCESSED_DIR" ]; then
    mkdir -p $PREPROCESSED_DIR
    echo "Created $PREPROCESSED_DIR"
fi

# FIXME: Need to find way to get iac-v1 and iac-v2 properly

python preprocess.py --dataset semeval-2018-irony

python preprocess.py --dataset riloff-sarcasm-data

python preprocess.py --dataset platek-sarcasm

# python preprocess.py --dataset iac-v1

# python preprocess.py --dataset iac-v2

python preprocess.py --dataset sarc-v2

python preprocess.py --dataset sarc-v2-pol

# filtered datasets, for

python preprocess.py --dataset riloff-sarcasm-data --min_len 5 --max_len 40

python preprocess.py --dataset platek-sarcasm --min_len 5 --max_len 40

# python preprocess.py --dataset iac-v1 --min_len 5 --max_len 80

# python preprocess.py --dataset iac-v2 --min_len 5 --max_len 80
