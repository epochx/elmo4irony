# elmo4irony: Deep contextualized word representations for detecting sarcasm and irony

This repo contains the official implementation of the paper "Deep contextualized word representations for detecting sarcasm and irony" WASSA 2018.

# Main Requirements

* Python 3
* Pytorch 0.4.0
* conda

# Installation

1. Clone this repo in your home directory

   ```bash
   git clone https://github.com/epochx/elmo4irony-dev
   cd elmo4irony
   ```

2. Create a conda environment.  If you don't have conda installed, we recommend using miniconda. You can then easily create and activate a new conda environment with Python 3.6 by executing:

   ```
   conda create -n elmo4irony python=3.6
   conda activate elmo4irony
   ```

3.  Run the installation script

    ```bash
    install/install.sh
    ```

If you want to clone the repo in a different place than your home directory, please check that the paths for the needed directories are properly set in [`config.py`](src/config.py).

# Data preparation

1.  Download the data (by default, to `~/data/elmo4irony/corpus`), by executing:

    ```bash
    download/download.sh
    ```

    > Some data need to be downloaded through the Twitter API. In order to do so
    > you need to apply for a Twitter developer account in the following link: 
    >
    > `https://developer.twitter.com/en/apply/user`
    >
    > Once you do so and create an app, fill the `.twitter_credentials.conf` file with
    > the `consumer_key`, `consumer_secret`, `access_token_key`, and
    > `access_token_secret` details.
    > 
    > Note 1: Downloading all the Twitter data will take around 24 hours.
    >
    > Note 2: During the download process the script will sleep due to some of the API's
    > restrictions.

2.  Prepare the data

    ```bash
    prepare/prepare.sh
    ```

3. Preprocess the data

    ```bash
    ./preprocess.sh
    ```

Finally, to test if you installed everything correctly, run:

```bash
python run.py --help
```

# Training

Run:

```
python run.py --corpus <corpus> --write_mode BOTH
```

to train a model with the default hyperparameters on the given `<corpus>`, and store the output results on disk. Checkpoints and other output files are saved in a directory named after the
`hash` of the current run in `~data/elmo4irony/results/`. 


> The `hash` will depend on hyperparameters that impact performance. For example, changing `learning_rate`, `lstm_hidden_size`, `dropout`, would produce different hashes, whereas changing `write_mode`, or `save_model` or similars, would not.

# Testing

To evaluate a trained model on the test set, run:

```
python run.py --corpus <corpus> --model_hash=<partial_model_hash> --test
```

Where you have to replace `<partial_model_hash>` by the hash of the model you
wish to test, corresponding to the name of its directory located in
`~data/elmo4irony/results/`. A classification report will be printed on screen, and files containing the test prediction labels  and probabilities ( `predictions.txt` and `test_probs.csv` respectively )will be created the model directory. Once you've run this, to obtain a more detailed output, you can can also try:

```
python evaluate.py --corpus <corpus> --predictions /path/to/predictions.txt
```
