#!/usr/bin/env bash

DATA_DIR=$1

cd "$DATA_DIR"
mkdir platek-sarcasm
cd platek-sarcasm

wget http://liks.fav.zcu.cz/sarcasm/en-balanced.zip
unzip en-balanced.zip

