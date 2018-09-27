import numpy as np


class Padder(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad1d(self, sequences, dim0_pad=None, dim1_pad=None,
              pad_lengths=False, align_right=False):
        """Pad a batch containing "1d" sequences

           Receive a list of sequences and return a padded 2d numpy ndarray,
           a numpy ndarray of lengths and a padded mask

           sequences: a list of lists, corresponding to sequences encoded in 1
                      hierarchical level, e.g. a sentence represented as a
                      sequence of words. The input `sequences` is a batch of such
                      sequences.

           len(sequences) = M, and N is the max sequence length contained in
           sequences.

            e.g.: [[2,45,3,23,54], [12,4,2,2], [4], [45, 12]]

           Return a numpy ndarray of dimension (M, N) corresponding to the
           padded sequence, a ndarray of the original lengths, and a mask.

           Returns:
               out: a numpy ndarray of dimension (M, N)
               lengths: a numpy ndarray of ints containing the lengths of each
                        input_list element
               mask: a numpy ndarray of dimension (M, N)
           """
        if not dim0_pad:
            dim0_pad = len(sequences)

        if not dim1_pad:
            dim1_pad = max(len(seq) for seq in sequences)

        out = np.full(shape=(dim0_pad, dim1_pad), fill_value=self.pad_id)
        mask = np.zeros(shape=(dim0_pad, dim1_pad))

        lengths = []
        for i in range(len(sequences)):
            data_length = len(sequences[i])
            ones = np.ones(data_length)
            lengths.append(data_length)
            offset = dim1_pad - data_length if align_right else 0
            np.put(out[i], range(offset, offset + data_length), sequences[i])
            np.put(mask[i], range(offset, offset + data_length),
                   np.ones(shape=(data_length)))

        lengths = np.array(lengths)
        return out, lengths, mask

    def pad2d(self, sequences2d, batch_first=True):
        """Pad a batch containing "2d" sequences

           sequences2d: A list containing lists of lists, corresponding to
               sequences encoded in 2 hierarchical levels, e.g. a sentence
               represented as a sequence of words represented as sequences of
               characters. The input `sequences2d` is a batch of such
               sequences.

           e.g.: [
                   [[1, 2, 3], [4, 5, 6]],
                   [[7, 8, 9, 10, 11], [1, 2, 5, 3, 6]],
                 ]

           return (padded_batch, first_h_lengths, second_h_lengths, masks) where
               padded_batch: 3d ndarray of dimension
                             (batch_size, max_sent_len, max_word_len)
               first_h_lengths: First hierarchy lengths. 1d ndarray of
                                dim (batch_size) corresponding to the first-level
                                hierarchy, e.g. sentence lengths
               second_h_lengths:second hierarchy lengths. 2d ndarray of
                                dim (batch_size, max_sent_len) corresponding to the
                                second level hierarchy, e.g. word lengths. Note
                                that this ndarray is padded with fake lengths of 1.
               masks: 3d ndarray with the same dim as padded_batch. This ndarrays
                      shows the positions of the valid items as opposed to the
                      paddings
           """
        if not batch_first:
            raise NotImplementedError

        batch_size = len(sequences2d)

        # TODO: rename variables make method more generic
        # max word length for the whole batch
        max_word_len = max([max(len(word) for word in char_sent)
                           for char_sent in sequences2d])

        max_sent_len = max([len(char_sent) for char_sent in sequences2d])

        # The second hierarchy lengths are padded with ones because a length of 0
        # makes no sense later in the process
        second_h_lengths = np.ones(shape=(batch_size, max_sent_len), dtype=np.int64)

        padded_batch = []
        first_h_lengths = []
        masks = []
        for i, sequence in enumerate(sequences2d):
            padded_sent, word_lengths, mask = self.pad1d(sequence,
                                                         dim0_pad=max_sent_len,
                                                         dim1_pad=max_word_len)

            np.put(second_h_lengths[i], range(len(word_lengths)), word_lengths)
            padded_batch.append(padded_sent)
            first_h_lengths.append(len(sequence))
            masks.append(mask)

        # -> (batch_size, max_sent_len, max_word_len)
        padded_batch = np.array(padded_batch)

        # -> (batch_size)
        first_h_lengths = np.array(first_h_lengths, dtype=np.int64)
        masks = np.array(masks)

        return padded_batch, first_h_lengths, second_h_lengths, masks

    def pad_vectors(self, sequences, dim0_pad=None, dim1_pad=None,
                    align_right=False):
        """Pad a batch containing lists of numpy ndarrays

           Receive a list of numpy ndarrays and return a padded 3d numpy array,
           a numpy ndarray of lengths and a padded mask

           sequences: a list of numpy ndarrays of shape (seq_len, emb_dim).
                      `sequences` will have len(sequences) of such ndarrays
           """
        raise NotImplementedError
        # Padding doesn't work properly. Only the last embedded vector is being
        # put in the corresponding first position of the batch dimension

        if not dim0_pad:
            dim0_pad = len(sequences)  # corresponding to the batch dimension
        if not dim1_pad:
            dim1_pad = max([seq.shape[0] for seq in sequences])

        dim2_pad = sequences[0].shape[1]

        assert all([dim2_pad == seq.shape[1] for seq in sequences]), "numpy arrays have different dimensions"

        out = np.full(shape=(dim0_pad, dim1_pad, dim2_pad),
                      fill_value=self.pad_id,
                      dtype=np.float32)
        mask = np.zeros(shape=(dim0_pad, dim1_pad, dim2_pad))

        lengths = []
        for i in range(len(sequences)):
            # seq: (seq_len, emb_dim)
            seq = sequences[i]
            seq_len = seq.shape[0]
            lengths.append(seq_len)
            offset = dim1_pad - seq_len if align_right else 0

            indices = [list(range(0, dim2_pad)) for i in range(offset, offset + seq_len)]
            indices = np.vstack(indices)  # (seq_len, emb_dim)

            np.put(out[i], indices, seq)

            np.put(mask[i], indices, np.ones(shape=(seq_len)))

        lengths = np.array(lengths)
        return out, lengths, mask
