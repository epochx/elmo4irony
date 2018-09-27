import os

HOME_DIR = os.environ['HOME']
ALLENNLP_PATH = os.path.join(HOME_DIR, 'allennlp')
DATA_PATH = os.path.join(HOME_DIR, 'data', 'elmo4irony')
CACHE_PATH = os.path.join(DATA_PATH, 'cache')
PREPARED_DATA_PATH = os.path.join(DATA_PATH, 'prepared')
PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'preprocessed')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')
LOG_PATH = os.path.join(RESULTS_PATH, 'log')


# EMBEDDINGS
EMBEDDINGS_DIR = os.path.join(DATA_PATH, 'word_embeddings')
ELMO_OPTIONS = os.path.join(EMBEDDINGS_DIR, 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
ELMO_WEIGHTS = os.path.join(EMBEDDINGS_DIR, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')

corpora_dict = {}
label_dict = {}

for corpus_name in os.listdir(PREPROCESSED_DATA_PATH):
    corpus_path = os.path.join(PREPROCESSED_DATA_PATH, corpus_name)
    if os.path.isdir(corpus_path):

        corpus_examples_dict = {
            'train': os.path.join(corpus_path, 'train.txt'),
            'dev': os.path.join(corpus_path, 'dev.txt'),
            'test': os.path.join(corpus_path, 'test.txt'),
        }

        corpus_labels_dict = {
            'train': os.path.join(corpus_path, 'train_labels.txt'),
            'dev': os.path.join(corpus_path, 'dev_labels.txt'),
            'test': os.path.join(corpus_path, 'test_labels.txt')
        }

        corpora_dict[corpus_name] = corpus_examples_dict
        label_dict[corpus_name] = corpus_labels_dict


WRITE_MODES = {'none': None,
               'file': 'FILE',
               'db': 'DATABASE',
               'both': 'BOTH'}

PAD_ID = 0
UNK_ID = 1
NUM_ID = 2
URL_ID = 3

# Specific to IEST dataset
USR_ID = 4
HASHTAG_ID = 5


PAD_TOKEN = '__PAD__'
UNK_TOKEN = '__UNK__'
NUM_TOKEN = '__NUM__'
URL_TOKEN = '__URL__'

# These should be just like the ones appearing in the input dataset (these are
# different to the originals because of preprocessing)
USR_TOKEN = '__USERNAME__'
HASHTAG_TOKEN = '__HASHTAG__'


SPECIAL_TOKENS = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        NUM_TOKEN: NUM_ID,
        URL_TOKEN: URL_ID,
        USR_TOKEN: USR_ID,
}

UNK_CHAR_ID = 0
UNK_CHAR_TOKEN = 'あ'

SPECIAL_CHARS = {
    UNK_CHAR_TOKEN: UNK_CHAR_ID,
}

LABEL2ID = {'0': 0, '1': 1}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}

LABELS = ['0', '1']

# DATABASE PARAMETERS
_DB_NAME = 'runs.db'

DATABASE_CONNECTION_STRING = 'sqlite:///' + os.path.join(RESULTS_PATH,
                                                         _DB_NAME)