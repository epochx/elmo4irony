from pad import pad1d, pad2d


def map_sequence(seq, sequence_map, unk_item_id):
    """ Transform a splitted sequence of items into another sequence of items
        according to the rules encoded in the dict item2id
       seq: iterable
       sequence_map: dict
       unk_item_id: int"""

    item_ids = []
    for item in seq:
        item_id = sequence_map.get(item, unk_item_id)
        item_ids.append(item_id)
    return item_ids


def map_sequences(sequences, sequence_map, unk_item_id):
    """Transform a list of sequences into another one, according to
       the rules encoded in sequence map"""

    mapped_sequences = []
    for seq in sequences:
        mapped_sequence = map_sequence(seq, sequence_map, unk_item_id)
        mapped_sequences.append(mapped_sequence)
    return mapped_sequences


def split_map_sequence(seq, sequence_map, unk_item_id, seq_splitter):
    """ Transform a sequence of items into another sequence of items
        according to the rules encoded in the dict item2id.

        Example usage: mapping words into their corresponding ids

       seq: iterable
       sequence_map: dict
       unk_item_id: int
       seq_splitter: function"""

    splitted_seq = seq_splitter(seq)
    item_ids = map_sequence(splitted_seq, sequence_map, unk_item_id)
    return item_ids


def split_map_sequences(sequences, sequence_map, unk_item_id, seq_splitter):
    """Split the sequences and then transform them into the items specified
       by sequence_map"""

    splitted_seqs = [seq_splitter(seq) for seq in sequences]
    splitted_mapped_seqs = map_sequences(splitted_seqs, sequence_map,
                                         unk_item_id)
    return splitted_mapped_seqs


def split_map_pad_sequences(sequences, sequence_map, unk_item_id, pad_id,
                            seq_splitter):
    """Split, transform (map) and pad a batch of sequences

       return the padded and mapped sequences, along with the original lengths
       and a mask indicating the real item positions, as opposed to the
       paddings"""

    splitted_mapped_sequences = split_map_sequences(
                          sequences, sequence_map, unk_item_id, seq_splitter)

    padded_mapped_sequences, lengths, mask = pad1d(
                          splitted_mapped_sequences, pad_id)

    return padded_mapped_sequences, lengths, mask


def split_sequences2d(sequences, seq_splitter_d1, seq_splitter_d2):
    """Split a sequence into its second level hierarchy components
       e.g. Split a string into its component words and characters.
    [
    'a brown cat sat on the red mat',
    'a gray fox jumped over the dog',
    'Phil saw Feel feel the feels'
    ]
    will become
    [
      [['a'], ['b', 'r', 'o', 'w', 'n'], ['c', 'a', 't'], ['s', 'a', 't'], ['o', 'n'], ['t', 'h', 'e'], ['r', 'e', 'd'], ['m', 'a', 't']],
      [['a'], ['g', 'r', 'a', 'y'], ['f', 'o', 'x'], ['j', 'u', 'm', 'p', 'e', 'd'], ['o', 'v', 'e', 'r'], ['t', 'h', 'e'], ['d', 'o', 'g']],
      [['P', 'h', 'i', 'l'], ['s', 'a', 'w'], ['F', 'e', 'e', 'l'], ['f', 'e', 'e', 'l'], ['t', 'h', 'e'], ['f', 'e', 'e', 'l', 's']]
    ]

       This will result in a doubly nested list"""
    splitted_seqs_d1 = [seq_splitter_d1(seqs) for seqs in sequences]
    splitted_seqs_d2 = []
    for splitted_seq_d1 in splitted_seqs_d1:
        splitted_seq_d2 = [seq_splitter_d2(seq_d2) for seq_d2
                           in splitted_seq_d1]
        splitted_seqs_d2.append(splitted_seq_d2)
    return splitted_seqs_d2


def split_map_sequences2d(sequences, sequence_map_d2, unk_item_id_d2,
                          seq_splitter_d1, seq_splitter_d2):
    """Split and transform (map) a batch of sequences into its second
       hierarchy level, e.g. convert a batch of strings into a batch of
       character-level-encoded sequences (words are the 1st hierarchy level,
       characters the 2nd one)

       [
       'a brown cat sat on the red mat',
        'a gray fox jumped over the dog',
        'Phil saw Feel feel the feels'
       ]
       will become
       [
         [[0], [1, 17, 14, 22, 13], [2, 0, 19], [18, 0, 19], [14, 13], [19, 7, 4], [17, 4, 3], [12, 0, 19]],
         [[0], [6, 17, 0, 24], [5, 14, 23], [9, 20, 12, 15, 4, 3], [14, 21, 4, 17], [19, 7, 4], [3, 14, 6]],
         [[99, 7, 8, 11], [18, 0, 22], [99, 4, 4, 11], [5, 4, 4, 11], [19, 7, 4], [5, 4, 4, 11, 18]]
       ]

       return the padded and mapped sequences, along with the original lengths
       and a mask indicating the real item positions, as opposed to the
       paddings"""

    splitted_seqs_d2 = split_sequences2d(sequences, seq_splitter_d1,
                                         seq_splitter_d2)
    splitted_mapped_seqs_d2 = []
    for splitted_seq_d2 in splitted_seqs_d2:
        splitted_mapped_sequences = map_sequences(splitted_seq_d2,
                                                  sequence_map_d2,
                                                  unk_item_id_d2)
        splitted_mapped_seqs_d2.append(splitted_mapped_sequences)
    return splitted_mapped_seqs_d2


def split_map_pad_sequences2d(sequences, sequence_map_d2, unk_item_id_d2,
                              pad_id_d2, seq_splitter_d1, seq_splitter_d2):

    splitted_mapped_seqs_d2 = split_map_sequences2d(
                            sequences, sequence_map_d2, unk_item_id_d2,
                            seq_splitter_d1, seq_splitter_d2)
    padded_batch, first_h_lengths, second_h_lengths, masks = \
        pad2d(splitted_mapped_seqs_d2, pad_id_d2)

    return padded_batch, first_h_lengths, second_h_lengths, masks


if __name__ == '__main__':

    seq = 'a cat sat on the red mat'
    splitted_seq = ['a', 'cat', 'sat', 'on', 'the', 'mat']
    sequence_map = {'cat': 1, 'mat': 2, 'a': 3, 'sat': 4, 'the': 5, 'on': 6,
                    'feel': 7, 'feels': 8, 'saw': 9}
    print(split_map_sequence(seq, sequence_map, 0, lambda x: x.split(' ')))
    print(map_sequence(splitted_seq, sequence_map, 0))

    print('Sequence map:\n', sequence_map)
    str_sequences = ['a brown cat sat on the red mat',
                     'a gray fox jumped over the dog',
                     'Phil saw Feel feel the feels']
    print('Sequences:\n', str_sequences)
    id_sequences = split_map_sequences(str_sequences, sequence_map, 0,
                                       lambda x: x.split(' '))
    print('Splitted and transformed sequences:\n',
          id_sequences)

    print('\n' + 72 * '#' + '\n')

    sequences = [[2, 45, 3, 23, 54], [12, 4, 2, 2], [4], [45, 12]]
    padded_sequences, lengths, mask = pad1d(sequences, 0)
    print('Original sequences:\n\t', sequences)
    print('Padded sequences:\n', padded_sequences)
    print('Lengths:\n', lengths)
    print('Mask:\n', mask)

    left_padded_sequences, lengths, left_padded_mask = \
        pad1d(sequences, 0, align_right=True)
    print('Left padded sequences:\n', left_padded_sequences)
    print('Left padded mask:\n', left_padded_mask)

    print('\n' + 72 * '#' + '\n')

    char_encoded_sent = [[[1, 2, 3], [4, 5, 6, 1], [10, 23], [3, 5, 2, 1, 76]],
                         [[7, 8, 9, 10, 11], [1, 2, 5, 3, 6, 10, 12]]]

    padded_batch, sentence_lengths, word_lengths, masks = \
        pad2d(char_encoded_sent, 0)

    print('Char-encoded sent:\n\t', char_encoded_sent)
    print('padded char-encoded sent:\n', padded_batch)
    print('sentence lengths:\n', sentence_lengths)
    print('word lengths tensor:\n', word_lengths)
    print('masks:\n', masks)

    print('\n' + 72 * '#' + '\n')

    print('Transform a batch of sentences into a padded batch of ids\n')
    print('Sequences:\n', str_sequences)
    padded_sequences, lengths, mask = split_map_pad_sequences(
                     str_sequences, sequence_map, 0, 0, lambda x: x.split(' '))
    print('Padded sequences:\n', padded_sequences)
    print('Lengths:\n', lengths)
    print('Mask:\n', mask)

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    sequence_map_d2 = {char: idx for idx, char in enumerate(alphabet)}
    splitted_seqs_d2 = split_sequences2d(str_sequences,
                                         lambda x: x.split(' '),
                                         lambda x: [y for y in x])
    print(splitted_seqs_d2)

    splitted_mapped_seqs_d2 = \
        split_map_sequences2d(str_sequences, sequence_map_d2, 99,
                              lambda x: x.split(' '),
                              lambda x: [y for y in x])
    print(splitted_mapped_seqs_d2)

    splitted_mapped_padded_seqs_d2 = \
        split_map_pad_sequences2d(
                            str_sequences, sequence_map_d2, 99,
                            33,
                            lambda x: x.split(' '),
                            lambda x: [y for y in x])

    print(splitted_mapped_padded_seqs_d2)
