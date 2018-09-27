
import numpy as np
import torch

from .. import config


def np_softmax(arr):
    """arr: numpy array of shape (N, D), the softmax will be applied on D"""
    # For numerical stability; softmax(x) = softmax(x - c) for any constant c
    # See https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    colwise_max = np.amax(arr, axis=1, keepdims=True)
    new_arr = arr - colwise_max

    exp = np.exp(new_arr)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs


def embed_context_window(embeddings, batch):
    """
    Return the embedded batch when using context windows

    embeddings: torch.nn.Embedding object
    dim batch: (seq_len, batch_size, window_size)

    return tensor of dimension (seq_len, batch_size, window_size*hidden)
    """
    assert len(batch.size()) == 3

    seq_len = batch.size(0)
    batch_size = batch.size(1)
    window_size = batch.size(2)
    hidden = embeddings.embedding_dim

    # -> (batch_size, seq_len, window_size)
    t_batch = batch.transpose(0, 1).contiguous()

    # -> (batch_size, seq_len*window_size)
    reshaped_t_batch = t_batch.view(batch_size, seq_len*window_size)

    # -> (batch_size, seq_len*window_size, hidden)
    emb_batch = embeddings(reshaped_t_batch)

    # -> (batch_size, seq_len, window_size*hidden)
    reshaped_emb_batch = emb_batch.view(batch_size, seq_len,
                                        window_size*hidden)

    # -> (seq_len, batch_size, window_size*hidden)
    t_emb_batch = reshaped_emb_batch.transpose(0, 1)

    return t_emb_batch


def distance(matrix1, matrix2, p=2):
    """Return the distance between the columns of two matrices.
       p represents the order of the distance p=2 is the Euclidean
       distance"""

    M = matrix1 - matrix2
    M2 = torch.pow(M, 2)
    M2_sum = torch.sum(M2, 1)
    return torch.sqrt(M2_sum)


def simple_columnwise_cosine_similarity(matrix1, matrix2):
    """Return the columnwise cosine similarity from matrix1 and matrix2.
    Expect matrices of dimension (batch_size, hidden).
    Return tensor of size (batch_size, hidden) containing the cosine
    similarities."""

    assert matrix1.size() == matrix2.size(), 'matrix sizes do not match'

    # -> (batch_size, 1)
    n_m1 = torch.norm(matrix1, 2, 1)
    n_m2 = torch.norm(matrix2, 2, 1)

    # -> (batch_size, 1)
    col_norm = torch.mul(n_m1, n_m2)

    # -> (batch_size, hidden)
    colprod = torch.mul(matrix1, matrix2)
    # -> (batch_size, 1)
    colsum = torch.sum(colprod, 1)

    # -> (batch_size, 1)
    cosine_sim = torch.div(colsum, col_norm)

    # -> (batch_size, 1)
    cosine_sim = cosine_sim.squeeze()

    return cosine_sim


def columnwise_cosine_similarity(tensor1, tensor2, dim=2, keep_dims=False,
                                 epsilon=1e-7):
    """Return the columnwise cosine similarity from tensor1 and tensor2.
    Expect tesor of dimension (batch_size, seq_len, hidden).
    Return tensor of size (batch_size, seq_len) containing the cosine
    similarities."""

    assert tensor1.size() == tensor2.size(), 'matrix sizes do not match'

    # -> (batch_size, seq_len, 1)
    n_m1 = torch.norm(tensor1, 2, dim)
    n_m2 = torch.norm(tensor2, 2, dim)

    # -> (batch_size, seq_len, 1)
    col_norm = torch.mul(n_m1, n_m2)

    # -> (batch_size, seq_len, hidden)
    colprod = torch.mul(tensor1, tensor2)
    # -> (batch_size, seq_len, 1)
    colsum = torch.sum(colprod, dim)

    # -> (batch_size, seq_len, 1)
    cosine_sim = torch.div(colsum, col_norm + epsilon)

    # -> (batch_size, seq_len)
    if not keep_dims:
        cosine_sim = cosine_sim.squeeze()

    return cosine_sim


def full_cosine_similarity(matrix1, matrix2):
    """
    Expect 2 matrices P and Q of dimension (d, n1) and (d, n2) respectively.
    Return a matrix A of dimension (n1, n2) with the result of comparing each
    vector to one another. A[i, j] represents the cosine similarity between
    vectors P[:, i] and Q[:, j].
    """
    n1 = matrix1.size(1)
    n2 = matrix2.size(1)
    d = matrix1.size(0)
    assert d == matrix2.size(0)

    # -> (d, n1, 1)
    t1 = matrix1.view(d, n1, 1)
    # -> (d, n1, n2)
    t1 = t1.repeat(1, 1, n2)

    # -> (d, 1, n2)
    t2 = matrix2.view(d, 1, n2)
    # -> (d, n1, n2)
    t2 = t2.repeat(1, n1, 1).contiguous()

    t1_x_t2 = torch.mul(t1, t2)  # (d, n1, n2)
    dotprod = torch.sum(t1_x_t2, 0).squeeze()  # (n1, n2)

    norm1 = torch.norm(t1, 2, 0)  # (n1, n2)
    norm2 = torch.norm(t2, 2, 0)  # (n1, n2)
    col_norm = torch.mul(norm1, norm2).squeeze()  # (n1, n2)

    return torch.div(dotprod, col_norm)  # (n1, n2)


def batch_full_cosine_similarity(tensor1, tensor2):
    """
    Expect 2 tensors tensor1 and tensor2 of dimension
    (batch_size, seq_len_p, hidden) and (batch_size, seq_len_q, hidden)
    respectively.

    Return a matrix A of dimension (batch_size, seq_len_p, seq_len_q) with the
    result of comparing each matrix to one another. A[k, :, :] represents the
    cosine similarity between matrices P[k, :, :] and Q[k, :, :]. Then
    A_k[i, j] is a scalar representing the cosine similarity between vectors
    P_k[i, :] and Q_k[j, :]
    """
    batch_size = tensor1.size(0)
    seq_len_p = tensor1.size(1)
    seq_len_q = tensor2.size(1)
    hidden = tensor1.size(2)
    assert batch_size == tensor2.size(0)
    assert hidden == tensor2.size(2)

    # -> (batch_size, seq_len_p, 1, hidden)
    t1 = tensor1.unsqueeze(2)
    # -> (batch_size, seq_len_p, seq_len_q, hidden)
    t1 = t1.repeat(1, 1, seq_len_q, 1)

    # -> (batch_size, 1, seq_len_q, hidden)
    t2 = tensor2.unsqueeze(1)
    # -> (batch_size, seq_len_p, seq_len_q, hidden)
    t2 = t2.repeat(1, seq_len_p, 1, 1)

    # -> (batch_size, seq_len_p, seq_len_q, hidden)
    t1_x_t2 = torch.mul(t1, t2)
    # -> (batch_size, seq_len_p, seq_len_q)
    dotprod = torch.sum(t1_x_t2, 3).squeeze(3)

    # norm1, norm2 and col_norm have dim (batch_size, seq_len_p, seq_len_q)
    norm1 = torch.norm(t1, 2, 3)
    norm2 = torch.norm(t2, 2, 3)
    col_norm = torch.mul(norm1, norm2).squeeze(3)

    return torch.div(dotprod, col_norm)  # (batch_size, seq_len_p, seq_len_q)


def mp_matching_op(v1, v2, W):
    """Compute the multi-perspective matching operation between
    two vectors and a perspective matrix.
    Both vectors are expected to be columns of dimension (d, 1),
    and the matrix W of dimension (l, d), where l are the
    perspectives and d the vector dimensions.

    Return a vector of size l containing the matching values"""
    l = W.size()[0]
    d = W.size()[1]
    assert d == v1.size()[0]
    assert d == v2.size()[0]
    assert len(v1.size()) == 2
    assert len(v2.size()) == 2

    # dim(v1)=(d, 1) -> (d, l)
    v1_broadcasted = v1.repeat(1, l)
    v2_broadcasted = v2.repeat(1, l)

    # dim(W)=(l, d) -> (d, l)
    W = W.transpose(1, 0)

    # elementwise multiplication
    Wv1 = torch.mul(W, v1_broadcasted)
    Wv2 = torch.mul(W, v2_broadcasted)

    return columnwise_cosine_similarity(Wv1, Wv2, dim=2)


def matrix_mp_matching_op(m1, m2, W):
    """Compute the multi-perspective matching operation between
    two tensors of column matrices and a perspective matrix.
    Both tensors are expected to be of dimension (batch_size, seq_len, hidden),
    and the matrix W of dimension (perspectives, hidden)

    Return a tensor of size (batch_size, perspectives, hidden)
    containing the matching values"""

    perspectives = W.size(0)
    hidden = W.size(1)
    seq_len = m1.size(1)
    batch_size = m1.size(0)
    assert batch_size == m2.size(0)
    assert hidden == m1.size(2)
    assert hidden == m2.size(2)
    assert m1.size(1) == m2.size(1)
    assert len(m1.size()) == 3

    m1 = m1.contiguous()
    m2 = m2.contiguous()

    # -> (batch_size, seq_len, hidden, 1)
    m1_exp = m1.unsqueeze(3)
    # -> (batch_size, seq_len, 1, hidden)
    m1_exp = m1_exp.transpose(2, 3)
    # -> (batch_size, seq_len, perspectives, hidden)
    m1_exp = m1_exp.repeat(1, 1, perspectives, 1).contiguous()
    # -> (batch_size, seq_len * perspectives, hidden)
    m1_exp = m1_exp.view(batch_size,
                         seq_len * perspectives,
                         hidden).contiguous()

    m2_exp = m2.unsqueeze(3).contiguous()
    m2_exp = m2_exp.transpose(2, 3)
    m2_exp = m2_exp.repeat(1, 1, perspectives, 1).contiguous()
    m2_exp = m2_exp.view(batch_size,
                         seq_len * perspectives,
                         hidden).contiguous()

    # -> (1, 1, perspectives, hidden)
    W = W.view(1, 1, perspectives, hidden)
    # -> (batch_size, seq_len, perspectives, hidden)
    W = W.repeat(batch_size, seq_len, 1, 1).contiguous()
    # -> (batch_size, seq_len * perspectives, hidden)
    W = W.view(batch_size, seq_len * perspectives, hidden)

    # -> (batch_size, seq_len * perspectives, hidden)
    Wm1 = torch.mul(W, m1_exp)
    Wm2 = torch.mul(W, m2_exp)

    # -> (batch_size, seq_len * perspectives)
    flat_sims = columnwise_cosine_similarity(Wm1, Wm2, dim=2)
    # -> (batch_size, seq_len, perspectives)
    sims = flat_sims.view(-1, seq_len, perspectives)
    # -> (batch_size, perspectives, seq_len)
    sims = sims.transpose(1, 2)
    return sims


def context_window(l, win, pad_id=config.PAD_ID):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [pad_id] + l + win // 2 * [pad_id]
    out = [lpadded[i:i + win] for i in range(len(l))]

    assert len(out) == len(l)

    return out
