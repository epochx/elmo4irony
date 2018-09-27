import os

# from profilehooks import profile

from ..utils.io import load_or_create
from .batch_iterator import BatchIterator
from .lang import Lang

from .. import config


class BaseCorpus(object):
    def __init__(self, paths_dict, corpus_name, use_chars=True,
                 force_reload=False, train_data_proportion=1.0,
                 dev_data_proportion=1.0, batch_size=64,
                 shuffle_batches=False, batch_first=True, lowercase=False):

        self.paths = paths_dict[corpus_name]
        self.corpus_name = corpus_name

        self.use_chars = use_chars

        self.force_reload = force_reload

        self.train_data_proportion = train_data_proportion
        self.dev_data_proportion = dev_data_proportion

        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.batch_first = batch_first

        self.lowercase = lowercase


class ClassificationCorpus(BaseCorpus):

    # @profile(immediate=True)
    def __init__(self, *args, use_pos=False, **kwargs):
        """args:
            paths_dict: a dict with two levels: <corpus_name>: <train/dev/rest>
            corpus_name: the <corpus_name> you want to use.

            We pass the whole dict containing all the paths for every corpus
            because it makes it easier to save and manage the cache pickles
        """
        super(ClassificationCorpus, self).__init__(*args, **kwargs)

        train_sents = open(self.paths['train']).readlines()
        self.use_pos = use_pos

        # This assumes the data come nicely separated by spaces. That's the
        # task of the tokenizer who should be called elsewhere
        self.train_sents = [s.rstrip().split() for s in train_sents]

        dev_sents = open(self.paths['dev']).readlines()
        self.dev_sents = [s.rstrip().split() for s in dev_sents]

        test_sents = open(self.paths['test']).readlines()
        self.test_sents = [s.rstrip().split() for s in test_sents]

        if self.lowercase:
            self.train_sents = [[t.lower() for t in s] for s in self.train_sents]
            self.dev_sents = [[t.lower() for t in s] for s in self.dev_sents]
            self.test_sents = [[t.lower() for t in s] for s in self.test_sents]

        lang_pickle_path = os.path.join(config.CACHE_PATH,
                                        self.corpus_name + '_lang.pkl')

        self.lang = load_or_create(lang_pickle_path,
                                   Lang,
                                   self.train_sents,
                                   force_reload=self.force_reload)

        self.label2id = {key: value for key, value in config.LABEL2ID.items()}

        self.train_examples = self._create_examples(
            self.train_sents,
            mode='train',
            prefix=self.corpus_name,
        )
        self.dev_examples = self._create_examples(
            self.dev_sents,
            mode='dev',
            prefix=self.corpus_name,
        )
        self.test_examples = self._create_examples(
            self.test_sents,
            mode='test',
            prefix=self.corpus_name,
        )

        if self.use_pos:
            pos_corpus = POSCorpus(config.pos_corpora_dict, self.corpus_name)
            self.pos_lang = pos_corpus.lang
            self._merge_pos_corpus(self.train_examples, pos_corpus.train_examples)
            self._merge_pos_corpus(self.dev_examples, pos_corpus.dev_examples)
            self._merge_pos_corpus(self.test_examples, pos_corpus.test_examples)

        self.train_batches = BatchIterator(
            self.train_examples,
            self.batch_size,
            data_proportion=self.train_data_proportion,
            shuffle=True,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
            use_pos=self.use_pos
        )

        self.dev_batches = BatchIterator(
            self.dev_examples,
            self.batch_size,
            data_proportion=self.dev_data_proportion,
            shuffle=False,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
            use_pos=self.use_pos
        )

        self.test_batches = BatchIterator(
            self.test_examples,
            self.batch_size,
            data_proportion=1.0,
            shuffle=False,
            batch_first=self.batch_first,
            use_chars=self.use_chars,
            use_pos=self.use_pos
        )

    @staticmethod
    def _merge_pos_corpus(examples, pos_examples):
        """Incorporate pos tag ids from POSCorpus
        examples : list of dicts
            The dataset examples
        pos_examples : list of dicts
            The POSCorpus examples
        Returns
        -------
        None
        Modifies the input examples

        """
        for i, example in enumerate(examples):

            # Sanity check. Make sure ids are the same.
            assert example['id'] == pos_examples[i]['id']
            # We need raw_sequences to have the same length as the pos tag sequences
            # To ensure this, run only one tokenizer in the preprocessing pipeline
            assert len(example['raw_sequence']) == len(pos_examples[i]['sequence'])

            example['pos_id_sequence'] = pos_examples[i]['sequence']

    def _create_examples(self, sents, mode, prefix):
        """
        sents: list of strings
        mode: (string) train, dev or test

        return:
            examples: a list containing dicts representing each example
        """

        allowed_modes = ['train', 'dev', 'test']
        if mode not in allowed_modes:
            raise ValueError(f'Mode not recognized, try one of {allowed_modes}')

        id_sents_pickle_path = os.path.join(
            config.CACHE_PATH,
            prefix + '_' + mode + '.pkl',
        )

        id_sents = load_or_create(id_sents_pickle_path,
                                  self.lang.sents2ids,
                                  sents,
                                  force_reload=self.force_reload)

        chars_pickle_path = os.path.join(
            config.CACHE_PATH,
            prefix + '_' + mode + '_chars.pkl',
        )

        char_id_sents = load_or_create(chars_pickle_path,
                                       self.lang.sents2char_ids,
                                       sents,
                                       force_reload=self.force_reload)

        #  FIXME: Assuming all 3 modes will have labels. This might not be the
        # case for test data <2018-06-29 10:49:29, Jorge Balazs>
        labels = open(config.label_dict[self.corpus_name][mode]).readlines()
        labels = [l.rstrip() for l in labels]
        id_labels = [self.label2id[label] for label in labels]

        ids = range(len(id_sents))

        examples = zip(ids,
                       sents,
                       id_sents,
                       char_id_sents,
                       id_labels)

        examples = [{'id': ex[0],
                     'raw_sequence': ex[1],
                     'sequence': ex[2],
                     'char_sequence': ex[3],
                     'label': ex[4]} for ex in examples]

        return examples


class POSCorpus(BaseCorpus):

    """A Corpus of POS tag features"""

    def __init__(self, *args, **kwargs):
        """
        We pass the whole dict containing all the paths for every corpus
        because it makes it easier to save and manage the cache pickles

        Parameters
        ----------
        paths_dict : dict
            a dict with two levels: <corpus_name>: <train/dev/rest>
        corpus_name : str
            the <corpus_name> you want to use.


        """
        super(POSCorpus, self).__init__(*args, **kwargs)
        self.corpus_name = "pos_" + self.corpus_name

        train_sents = open(self.paths['train']).readlines()

        # This assumes the data come nicely separated by spaces. That's the
        # task of the tokenizer who should be called elsewhere
        self.train_sents = [s.rstrip().split() for s in train_sents]

        dev_sents = open(self.paths['dev']).readlines()
        self.dev_sents = [s.rstrip().split() for s in dev_sents]

        test_sents = open(self.paths['test']).readlines()
        self.test_sents = [s.rstrip().split() for s in test_sents]

        lang_pickle_path = os.path.join(config.CACHE_PATH,
                                        self.corpus_name + '_lang.pkl')

        all_sents = self.train_sents + self.dev_sents + self.test_sents
        self.lang = load_or_create(lang_pickle_path,
                                   Lang,
                                   all_sents,
                                   force_reload=self.force_reload)

        self.train_examples = self._create_examples(
            self.train_sents,
            mode='train',
            prefix=self.corpus_name,
        )
        self.dev_examples = self._create_examples(
            self.dev_sents,
            mode='dev',
            prefix=self.corpus_name,
        )
        self.test_examples = self._create_examples(
            self.test_sents,
            mode='test',
            prefix=self.corpus_name,
        )

    def _create_examples(self, sents, mode, prefix):
        """Creates the examples container for POSCorpus

        Parameters
        ----------
        sents : list
            a list of lists containing pos tags
        mode : str {train, dev, test}
            for cache naming purposes
        prefix : str
            for cache naming purposes

        Returns
        -------
        list of dicts representing each example

        """
        ids = range(len(sents))
        # process could be slightly optimized by caching the results of the line
        # below
        pos_ids = self.lang.sents2ids(sents)
        examples = zip(ids, sents, pos_ids)

        examples = [{'id': ex[0],
                     'raw_sequence': ex[1],
                     'sequence': ex[2]} for ex in examples]

        for example in examples:
            assert len(example['raw_sequence']) == len(example['sequence'])

        return examples
