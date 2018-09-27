#!/usr/bin/env bash

TOPLEVEL_DIR=$PWD
DATA_DIR="$HOME/data/elmo4irony/corpus"
EMBEDDINGS_DIR="$HOME/data/elmo4irony/word_embeddings"

if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    echo "Created $DATA_DIR"
fi

sh download/download_platek.sh "$DATA_DIR"

sh download/download_riloff.sh "$DATA_DIR"

sh download/download_sarc_v2.sh "$DATA_DIR"

sh download/download_sarc_v2_pol.sh "$DATA_DIR"

sh download/download_semeval_2018.sh "$DATA_DIR"


# Get ELMo data
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    mkdir -p "$EMBEDDINGS_DIR"
    echo "Created $EMBEDDINGS_DIR"
fi

cd "$EMBEDDINGS_DIR"
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
cd "$TOPLEVEL_DIR"


# Crawl tweets
echo "Getting Twitter data, this will take around 24 hours"
echo
python download/twitter_api_crawler.py --input "$DATA_DIR/platek-sarcasm/en-balanced/normal.txt"
python download/twitter_web_crawler.py --input "$DATA_DIR/platek-sarcasm/en-balanced/sarcastic.txt"

awk '{print $1}' "$DATA_DIR/riloff-sarcasm-data/sarcasm-data/sarcasm-annos-emnlp13.tsv" \
    > "$DATA_DIR/riloff-sarcasm-data/sarcasm-annos-emnlp13.txt"

python download/twitter_api_crawler.py --input "$DATA_DIR/riloff-sarcasm-data/sarcasm-annos-emnlp13.txt"
