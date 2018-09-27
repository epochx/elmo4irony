#!/usr/bin/env bash

tail -n +2 irony_wassa.csv | awk -F',' '{print $1}' | sed 's\"\\g' > irony_wassa_clean.txt

tail -n +2 dump_wassa.csv | awk -F',' '{print $1}' | sed 's\"\\g' > dump_wassa_clean.txt

tail -n +2 sarcasm_wassa.csv | awk -F',' '{print $1}' | sed 's\"\\g' > sarcasm_wassa_clean.txt