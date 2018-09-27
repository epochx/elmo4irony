#!/usr/bin/env bash


python preprocess.py --dataset semeval-2018-irony \
    --augment ~/data/elmo4irony/prepared/semeval-2018-irony-augmented_ironic.txt \
              ~/data/elmo4irony/prepared/semeval-2018-irony-augmented_nonironic.txt \
    --name semeval-2018-irony-augmented

python preprocess.py --dataset riloff-sarcasm-data \
    --augment ~/data/elmo4irony/prepared/riloff-sarcasm-data-augmented_ironic.txt \
              ~/data/elmo4irony/prepared/riloff-sarcasm-data-augmented_nonironic.txt \
    --name riloff-sarcasm-data-augmented

python preprocess.py --dataset platek-sarcasm \
    --augment ~/data/elmo4irony/prepared/platek-sarcasm-augmented_ironic.txt \
              ~/data/elmo4ironyyapo/prepared/platek-sarcasm-augmented_nonironic.txt \
    --name platek-sarcasm-augmented
