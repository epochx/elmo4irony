
import torch

from torch import nn
from ..layers.pooling import PoolingLayer
from ..layers.elmo import ElmoWordEncodingLayer
from ..utils.torch import to_var, pack_forward


class WordEncodingLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEncodingLayer, self).__init__()
        self.word_encoding_method = 'elmo'
        self.word_encoding_layer = ElmoWordEncodingLayer(**kwargs)
        self.embedding_dim = self.word_encoding_layer.embedding_dim

    def __call__(self, *args, **kwargs):
        return self.word_encoding_layer(*args, **kwargs)

    def __repr__(self):
        s = '{name}('
        s += 'method={word_encoding_method}'
        s += ')'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

class BLSTMEncoder(nn.Module):
    """
    Args:
        embedding_dim: """

    def __init__(self, embedding_dim, hidden_sizes=2048, num_layers=1,
                 bidirectional=True, dropout=0.0, batch_first=True,
                 use_cuda=True):
        super(BLSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_sizes  # sizes in plural for compatibility
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.dropout = dropout
        self.batch_first = batch_first
        self.out_dim = self.hidden_size * self.num_dirs

        self.enc_lstm = nn.LSTM(self.embedding_dim, self.hidden_size,
                                num_layers=self.num_layers, bidirectional=True,
                                dropout=self.dropout)

    def is_cuda(self):
        # FIXME: Avoid calls to data()
        # either all weights are on cpu or they are on gpu
        return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

    def forward(self, emb_batch, lengths=None, masks=None):
        """mask kept for compatibility with transformer layer"""
        sent_output = pack_forward(self.enc_lstm, emb_batch, lengths)
        return sent_output


class SentenceEncodingLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(SentenceEncodingLayer, self).__init__()
        self.sent_encoding_method = 'bilstm'
        self.sent_encoding_layer = BLSTMEncoder(*args, **kwargs)
        self.out_dim = self.sent_encoding_layer.out_dim

    def __call__(self, *args, **kwargs):
        return self.sent_encoding_layer(*args, **kwargs)


class Classifier(nn.Module):
    """ Classifier

    Parameters
    ----------
    num_classes : int
    batch_size : int
    sent_encoding_method : str
    hidden_sizes : list
    sent_enc_layers : int
    pooling_method : str
    batch_first : bool
    dropout : float
    sent_enc_dropout : float
    use_cuda : bool
        """
    def __init__(self,
                 num_classes,
                 batch_size,
                 hidden_sizes=None,
                 sent_enc_layers=1,
                 pooling_method='max',
                 batch_first=True,
                 dropout=0.0,
                 sent_enc_dropout=0.0,
                 use_cuda=True):

        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.sent_enc_dropout = sent_enc_dropout

        self.use_cuda = use_cuda

        self.pooling_method = pooling_method
        self.hidden_sizes = hidden_sizes
        self.sent_enc_layers = sent_enc_layers

        self.word_encoding_layer = WordEncodingLayer(use_cuda=self.use_cuda)

        self.word_dropout = nn.Dropout(self.dropout)

        self.sent_encoding_layer = SentenceEncodingLayer(
            self.word_encoding_layer.embedding_dim,
            hidden_sizes=self.hidden_sizes,
            num_layers=self.sent_enc_layers,
            batch_first=self.batch_first,
            use_cuda=self.use_cuda,
            dropout=self.sent_enc_dropout
        )

        self.sent_dropout = nn.Dropout(self.sent_enc_dropout)

        sent_encoding_dim = self.sent_encoding_layer.out_dim

        self.pooling_layer = PoolingLayer(self.pooling_method,
                                          sent_encoding_dim)

        self.dense_layer = nn.Sequential(
            nn.Linear(self.pooling_layer.out_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes)
        )

    def encode(self, batch, sent_lengths,  masks=None,
               char_masks=None, embed_words=True, raw_sequences=None):
        """ Encode a batch of ids into a sentence representation.

            This method exists for compatibility with facebook's senteval

            batch: padded batch of word indices if embed_words, else padded
                   batch of torch tensors corresponding to embedded word
                   vectors

            embed_words: whether to pass the input through an embedding layer
                         or not
        """

        embedded = self.word_encoding_layer(raw_sequences)

        # elmo_masks are needed when masking the output from the ELMo
        # layer. When using the BiLSTM after ELMo we use the sentence
        # lengths (obtained from the id sequences) to pack the
        # sequences. This makes the output of the BiLSTM compatible with
        # the masks generated from the id sequences. When trying to mix
        # both we get dimension mismatch errors.
        embedded, elmo_masks = embedded
        elmo_masks = elmo_masks.float()
        masks = elmo_masks

        embedded = self.word_dropout(embedded)

        sent_embedding = self.sent_encoding_layer(embedded,
                                                  lengths=sent_lengths,
                                                  masks=masks)

        sent_embedding = self.sent_dropout(sent_embedding)

        agg_sent_embedding = self.pooling_layer(sent_embedding,
                                                lengths=sent_lengths,
                                                masks=masks)
        return agg_sent_embedding

    def forward(self, batch):

        # batch is created in Batch Iterator
        sequences = batch['sequences']
        raw_sequences = batch['raw_sequences']

        # when using raw_sequence_lengths (that consider periods within
        # sentences) we also need to use the elmo masks
        sent_lengths = batch['raw_sequence_lengths']

        masks = batch['masks']

        # TODO: to_var is going to happen for every batch every epoch which
        # makes this op O(num_batches * num_epochs). We could make it
        # O(num_batches) if we ran it once for every batch before training, but
        # this would limit our ability to shuffle the examples and re-create
        # the batches each epoch
        sent_lengths = to_var(torch.FloatTensor(sent_lengths),
                              self.use_cuda,
                              requires_grad=False)
        masks = to_var(torch.FloatTensor(masks),
                       self.use_cuda,
                       requires_grad=False)

        sent_vec = self.encode(sequences,
                               sent_lengths=sent_lengths,
                               masks=masks,
                               raw_sequences=raw_sequences)

        logits = self.dense_layer(sent_vec)

        ret_dict = {'logits': logits}

        return ret_dict
