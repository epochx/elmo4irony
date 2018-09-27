import random

import numpy as np

from .. import config
from ..seq_utils.pad import Padder

random.seed(1111)


class BaseNLPBatch(dict):
    def __init__(self, *args, **kwargs):
        super(BaseNLPBatch, self).__init__(*args, **kwargs)

        self.batch_first = kwargs.pop('batch_first')
        self.batch_size = kwargs.pop('batch_size')
        self.padder = Padder(config.PAD_ID)
        self.use_pos = kwargs.pop('use_pos')

    def _pad1d(self, sequences, *args, **kwargs):
        """sequences: a list of lists"""
        padded_sequences, lengths, masks = self.padder.pad1d(sequences,
                                                             *args,
                                                             **kwargs)
        if not self.batch_first:
            padded_sequences = padded_sequences.transpose(1, 0)
            masks = masks.transpose(1, 0)

        return padded_sequences, lengths, masks

    def _pad2d(self, sequences2d, *args, **kwargs):
        """sequences2d: a list of lists of lists"""
        (padded_sequences2d,
         sent_lengths,
         word_lengths,
         char_masks) = self.padder.pad2d(sequences2d, *args, **kwargs)
        if not self.batch_first:
            padded_sequences2d = padded_sequences2d.transpose(1, 2, 0)
            char_masks = char_masks.transpose(1, 2, 0)

        return padded_sequences2d, sent_lengths, word_lengths, char_masks


class IESTBatch(BaseNLPBatch):
    def __init__(self, examples, *args, **kwargs):
        super(IESTBatch, self).__init__(*args, **kwargs)
        self.examples = examples
        self.use_chars = kwargs.pop('use_chars')
        self._build_batch_from_examples()

    def _build_batch_from_examples(self):

        # This class expects examples to be a list containing dicts
        # with at least a 'sequence', a 'label' key and a 'char_sequence'
        # if use_chars is true
        ids = [example['id'] for example in self.examples]

        sequences = [example['sequence'] for example in self.examples]
        padded_sequences, sent_lengths, masks = self._pad1d(sequences)

        self['raw_sequences'] = [example['raw_sequence'] for example in self.examples]
        self['raw_sequence_lengths'] = np.array([len(example['raw_sequence'])
                                                 for example in self.examples])
        self['sequences'] = padded_sequences
        self['sent_lengths'] = sent_lengths
        self['masks'] = masks

        labels = [example['label'] for example in self.examples]
        self['labels'] = labels

        self['ids'] = ids

        if self.use_pos:
            pos_sequences = [example['pos_id_sequence'] for example in self.examples]
            self['pos_sequences'], _, _ = self._pad1d(pos_sequences)

        if self.use_chars:
            char_sequences = [example['char_sequence']
                              for example in self.examples]

            (padded_sequences2d,
             sent_lengths,
             word_lengths,
             char_masks) = self._pad2d(char_sequences)

            self['char_sequences'] = padded_sequences2d
            self['word_lengths'] = word_lengths
            self['char_masks'] = char_masks

    def inspect(self):
        for key, value in self.items():
            try:
                print(f'{key}: shape={value.shape}')
            except AttributeError:
                if isinstance(value, list):
                    print(f'{key}: length={len(value)}')
                elif isinstance(value, bool) or isinstance(value, int):
                    print(f'{key}: {value}')
                else:
                    print(f'{key}: type={type(value)}')

    def __repr__(self):
        return self.__class__.__name__


class BatchIterator(object):

    def __init__(self, examples, batch_size, data_proportion=1.0,
                 shuffle=False, batch_first=False, use_chars=False,
                 use_pos=False):

        """Create batches of length batch_size from the examples
        Args:
            examples: The data to be batched. Independent from corpus or model
            batch_size: The desired batch size.
            shuffle: whether to shuffle the data before creating the batches.
            batch_first:
        """

        self.examples = examples
        self.batch_size = batch_size

        self.data_proportion = data_proportion
        self.shuffle = shuffle
        self.batch_first = batch_first

        self.use_chars = use_chars
        self.use_pos = use_pos

        if shuffle:
            random.shuffle(self.examples)

        self.examples_subset = self.examples

        assert 0.0 < data_proportion <= 1.0
        self.n_examples_to_use = int(len(self.examples_subset) *
                                     self.data_proportion)

        self.examples_subset = self.examples_subset[:self.n_examples_to_use]

        self.num_batches = (self.n_examples_to_use +
                            batch_size - 1) // batch_size

        self.labels = []
        self.ids = []

        self.num_batches = (len(self.examples_subset) + batch_size - 1) // batch_size

    def __getitem__(self, index):
        assert index < self.num_batches, ("Index is greater "
                                          "than the number of batches "
                                          "%d>%d" % (index, self.num_batches))

        # First we obtain the batch slices
        examples_slice = self.examples_subset[index * self.batch_size:
                                              (index + 1) * self.batch_size]

        return IESTBatch(examples_slice,
                         batch_size=self.batch_size,
                         batch_first=self.batch_first,
                         use_chars=self.use_chars,
                         use_pos=self.use_pos)

    def __len__(self):
        return self.num_batches

    def shuffle_examples(self):
        random.shuffle(self.examples_subset)
