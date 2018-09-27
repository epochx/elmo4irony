#!/usr/bin/env bash

awk -F'\t' '{print $2}' SemEval2018-AIT-DISC.tweets.txt | uniq > SemEval2018-AIT-DISC.tweets.clean.txt