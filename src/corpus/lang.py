# import os
from collections import defaultdict, OrderedDict

from .. import config
from ..utils.io import load_or_create


class Lang(object):

    def __init__(self, sents,  min_freq_threshold=0, force_reload=False):
        """sents is a list of tokenized sentences"""

        # token_dict = load_or_create(token_dict_pickle_path,
        #                             build_word_lang,
        #                             sents,
        #                             min_freq_threshold=min_freq_threshold,
        #                             force_reload=force_reload)
        self.token2id, self.token_freqs = self._build_token_dict(sents)

        # char_dict = load_or_create(char_dict_pickle_path,
        #                            build_char_lang,
        #                            sents,
        #                            force_reload=force_reload)

        self.char2id, self.char_freqs = self._build_char_dict()

    def _build_token_dict(self, sents):
        #  FIXME: Hardcoding the usage of the SPECIAL_TOKENS global here makes
        # it difficult to use this class for purposes other than processing NLP
        # sentences. For example I'm using this for parsing POS tags that come
        # arranged in the same format as the input data, and the special tokens
        # are being preprended to the the token_dict even though they make
        # no sense there. The solution would be to pass the special tokens and
        # special chars to the class constructor and work with those
        # <2018-07-05 12:06:53, Jorge Balazs>
        token_dict = {key: value for key, value in config.SPECIAL_TOKENS.items()}
        curr_token_id = len(token_dict)
        token_freqs = defaultdict(int)
        for sent in sents:
            for token in sent:
                token_freqs[token] += 1
                if token not in token_dict.keys():
                    token_dict[token] = curr_token_id
                    curr_token_id += 1

        token_freqs = OrderedDict(sorted(token_freqs.items(),
                                         key=lambda t: t[1],
                                         reverse=True))
        return token_dict, token_freqs

    def _build_char_dict(self):
        char_dict = {key: value for key, value in config.SPECIAL_CHARS.items()}
        curr_char_id = len(char_dict)
        char_freqs = defaultdict(int)

        for token in self.token2id.keys():
            if token not in config.SPECIAL_TOKENS.keys():
                for char in token:
                    char_freqs[char] += 1
                    if char not in char_dict.keys():
                        char_dict[char] = curr_char_id
                        curr_char_id += 1

        char_freqs = OrderedDict(sorted(char_freqs.items(),
                                        key=lambda t: t[1],
                                        reverse=True))

        return char_dict, char_freqs

    def sent2ids(self, sent, ignore_period=True, append_EOS=False):
        if not isinstance(sent, list):
            raise TypeError(f'Input shout be a list but got {type(sent)} instead.')
        ids = []
        for token in sent:
            if token == '.' and ignore_period:
                continue
            try:
                ids.append(self.token2id[token])
            except KeyError:
                ids.append(config.UNK_ID)
        if append_EOS:
            ids.append(config.EOS_ID)
        return ids

    def sents2ids(self, sents, ignore_period=True, append_EOS=False):
        if not isinstance(sents, list):
            raise TypeError(f'Expected list but got {type(sents)} instead.')
        id_sents = []
        for sent in sents:
            id_sents.append(self.sent2ids(sent,
                                          ignore_period=ignore_period,
                                          append_EOS=append_EOS))
        return id_sents

    def token2char_ids(self, token):
        if not isinstance(token, str):
            raise TypeError(f'Input shout be a str but got {type(token)} instead.')

        # FIXME: token here will never be UNK_TOKEN, because UNKs have not been
        # defined for characters previous to this step
        if token in (config.UNK_TOKEN, config.NUM_TOKEN, config.URL_TOKEN):
            return [config.UNK_CHAR_ID]

        char_ids = []
        for char in token:
            try:
                char_ids.append(self.char2id[char])
            except KeyError:
                char_ids.append(config.UNK_CHAR_ID)
        return char_ids

    def sent2char_ids(self, sent, ignore_period=True):
        if not isinstance(sent, list):
            raise TypeError(f'Input shout be a list but got {type(sent)} instead.')
        char_ids = []

        for token in sent:
            if token == '.' and ignore_period:
                continue
            char_ids.append(self.token2char_ids(token))
        return char_ids

    def sents2char_ids(self, sents, ignore_period=True):
        if not isinstance(sents, list):
            raise TypeError(f'Expected list but got {type(sents)} instead.')

        sent_char_ids = []
        for sent in sents:
            sent_char_ids.append(self.sent2char_ids(sent,
                                                    ignore_period=ignore_period))
        return sent_char_ids
