#!/usr/bin/env bash

DATA_DIR=$1

cd "$DATA_DIR"


wget http://nlp.cs.princeton.edu/SARC/2.0/pol/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/train-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/test-balanced.csv.bz2

bunzip comments.json.bz2
bunzip train-balanced.csv.bz2
bunzip test-balanced.csv.bz2