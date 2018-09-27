#!/bin/bash

conda install -y pytorch=0.4.0 cuda80 -c pytorch
conda install -y scikit-learn unzip
conda install -y requests
pip install -r requirements.txt

cd "$HOME"

git clone https://github.com/allenai/allennlp.git
cd allennlp
# Using exactly the same version we used for development
git checkout ac2e0b9b6e4668984ebd8c05578d9f4894e94bee
INSTALL_TEST_REQUIREMENTS=false scripts/install_requirements.sh
