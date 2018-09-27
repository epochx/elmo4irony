#!/usr/bin/env bash

DATA_DIR="$HOME/data/corpus"
PREPARED_DIR="$HOME/data/elmo4irony/prepared"

if [ ! -d "$PREPARED_DIR" ]; then
    mkdir -p $PREPARED_DIR
    echo "Created $PREPARED_DIR"
fi

cd "$PREPARED_DIR"

python "prepare/prepare_semeval_2018_irony.py" "$DATA_DIR/SemEval2018-Task3-master"

python "prepare/prepare_iac_v1.py" "$DATA_DIR/IAC/data-sarc-sample"

python "prepare/prepare_iac_v2.py" "$DATA_DIR/IAC/sarcasm_v2.csv"

python "prepare/prepare_riloff.py" "$DATA_DIR/riloff-sarcasm-data"

python "prepare/prepare_platek.py" "$DATA_DIR/platek-sarcasm"

python "prepare/prepare_sarc_v2.py" "$DATA_DIR/SARC2.0"

python "prepare/prepare_sarc_v2_pol.py" "$DATA_DIR/SARC2.0"
