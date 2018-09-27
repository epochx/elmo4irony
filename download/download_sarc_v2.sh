#!/usr/bin/env bash

DATA_DIR=$1

cd "$DATA_DIR"

mkdir -p SARC2.0/main
cd SARC2.0/main

wget http://nlp.cs.princeton.edu/SARC/2.0/main/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/train-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/test-balanced.csv.bz2

bzip2 -dk comments.json.bz2
bzip2 -dk train-balanced.csv.bz2
bzip2 -dk test-balanced.csv.bz2

