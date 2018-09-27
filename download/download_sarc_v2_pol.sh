#!/usr/bin/env bash

DATA_DIR=$1
SARC_POL_DIR="$DATA_DIR/SARC2.0/pol"

if [ ! -d "$SARC_POL_DIR" ]; then
    mkdir -p $SARC_POL_DIR
    echo "Created $SARC_POL_DIR"
fi

cd "$SARC_POL_DIR"

wget http://nlp.cs.princeton.edu/SARC/2.0/pol/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/train-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/test-balanced.csv.bz2

bzip2 -dk comments.json.bz2
bzip2 -dk train-balanced.csv.bz2
bzip2 -dk test-balanced.csv.bz2
