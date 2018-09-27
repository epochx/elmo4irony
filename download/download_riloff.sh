#!/usr/bin/env bash

DATA_DIR=$1

cd "$DATA_DIR"

mkdir riloff-sarcasm-data
cd riloff-sarcasm-data

wget http://www.cs.utah.edu/~riloff/sarcasm-data/sarcasm-data.tar.gz
tar -xvzf sarcasm-data.tar.gz