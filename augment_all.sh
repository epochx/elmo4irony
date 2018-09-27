#!/usr/bin/env bash

cd ~/data/chandler/prepared

python ~/chandler/augment_prepared.py --dataset semeval-2018-irony \
    --ironic ~/data/corpus/ironsarc/irony_wassa_clean.txt ~/data/corpus/ironsarc/sarcasm_wassa_clean.txt \
    --nonironic ~/data/corpus/SemEval2018-AIT-DISC/SemEval2018-AIT-DISC.tweets.clean.txt ~/data/corpus/ironsarc/dump_wassa_clean.txt \
    --name semeval-2018-irony-augmented

python ~/chandler/augment_prepared.py --dataset riloff-sarcasm-data \
    --ironic ~/data/corpus/ironsarc/irony_wassa_clean.txt ~/data/corpus/ironsarc/sarcasm_wassa_clean.txt \
    --nonironic ~/data/corpus/SemEval2018-AIT-DISC/SemEval2018-AIT-DISC.tweets.clean.txt ~/data/corpus/ironsarc/dump_wassa_clean.txt \
    --name riloff-sarcasm-data-augmented

python ~/chandler/augment_prepared.py --dataset platek-sarcasm \
    --ironic ~/data/corpus/ironsarc/irony_wassa_clean.txt ~/data/corpus/ironsarc/sarcasm_wassa_clean.txt \
    --nonironic ~/data/corpus/SemEval2018-AIT-DISC/SemEval2018-AIT-DISC.tweets.clean.txt ~/data/corpus/ironsarc/dump_wassa_clean.txt \
    --name platek-sarcasm-augmented