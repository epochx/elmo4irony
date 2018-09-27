#!/usr/bin/env bash

DATA_DIR=$1

cd "$DATA_DIR"

mkdir -p SARC2.0/main
cd SARC2.0/main

wget http://nlp.cs.princeton.edu/SARC/2.0/main/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/train-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/test-balanced.csv.bz2

bunzip comments.json.bz2
bunzip train-balanced.csv.bz2
bunzip test-balanced.csv.bz2

