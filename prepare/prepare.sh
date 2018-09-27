#!/usr/bin/env bash

DATA_DIR="$HOME/data/elmo4irony/corpus"
PREPARED_DIR="$HOME/data/elmo4irony/prepared"

if [ ! -d "$PREPARED_DIR" ]; then
    mkdir -p $PREPARED_DIR
    echo "Created $PREPARED_DIR"
fi

echo "Preparing SemEval 2018 dataset..."
python "prepare/prepare_semeval_2018_irony.py" "$DATA_DIR/SemEval2018-Task3-master" "$PREPARED_DIR"

echo "Preparing Riloff dataset..."
python "prepare/prepare_riloff.py" "$DATA_DIR/riloff-sarcasm-data" "$PREPARED_DIR"

echo "Preparing Platek dataset..."
python "prepare/prepare_platek.py" "$DATA_DIR/platek-sarcasm" "$PREPARED_DIR"

echo "Preparing SARC2.0 dataset..."
python "prepare/prepare_sarc_v2.py" "$DATA_DIR/SARC2.0" "$PREPARED_DIR"

echo "Preparing SARC2.0-pol dataset..."
python "prepare/prepare_sarc_v2_pol.py" "$DATA_DIR/SARC2.0" "$PREPARED_DIR"


# FIXME: Need to find way to get these datasets properly
# If you uncomment the lines below, remember to also do it in preprocess.sh

# python "prepare/prepare_iac_v1.py" "$DATA_DIR/IAC/data-sarc-sample" "$PREPARED_DIR"
# python "prepare/prepare_iac_v2.py" "$DATA_DIR/IAC/sarcasm_v2.csv" "$PREPARED_DIR"
