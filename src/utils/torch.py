import logging

import torch
from torch import nn

import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def to_var(tensor, use_cuda, requires_grad=True):
    """Transform tensor into variable and transfer to GPU depending on flag"""
    if requires_grad:
        tensor.requires_grad_()

    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def pack_forward(module, emb_batch, lengths, use_cuda=True, batch_first=True):
    """Based on: https://github.com/facebookresearch/InferSent/blob/4b7f9ec7192fc0eed02bc890a56612efc1fb1147/models.py

       Automatically sort and pck a padded sequence, feed it to an RNN and then
       unpack, re-pad and unsort it.

        Args:
            module: an instance of a torch RNN module
            batch: a pytorch tensor of dimension (batch_size, seq_len, emb_dim)
            lengths: a pytorch tensor of dimension (batch_size)"""

    sent = emb_batch
    # FIXME: should avoid calls do data() since pytorch 0.4.0
    sent_len = lengths.data.cpu().numpy()

    # Sort by length (keep idx)
    sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
    idx_unsort = np.argsort(idx_sort)

    idx_sort = torch.from_numpy(idx_sort).cuda() if use_cuda \
        else torch.from_numpy(idx_sort)
    # idx_sort = Variable(idx_sort)
    sent = sent.index_select(0, idx_sort)

    sent_len_list = sent_len.tolist()
    sent_len_list = [int(elem) for elem in sent_len_list]

    # Handling padding in Recurrent Networks
    sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_list,
                                                    batch_first=batch_first)

    sent_output, _ = module(sent_packed)

    sent_output = nn.utils.rnn.pad_packed_sequence(sent_output,
                                                   batch_first=batch_first,
                                                   padding_value=config.PAD_ID)[0]

    # Un-sort by length
    idx_unsort = torch.from_numpy(idx_unsort).cuda() if use_cuda \
        else torch.from_numpy(idx_unsort)
    # idx_unsort = Variable(idx_unsort)

    select_dim = 0 if batch_first else 1
    sent_output = sent_output.index_select(select_dim, idx_unsort)

    return sent_output


import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.
    From https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_free_gpu_index(max_memory=10, unallowed_gpus=None):
    """Get a random GPU with at most max_memory used"""

    if unallowed_gpus is None:
        unallowed_gpus = []

    gpu_memory_map = get_gpu_memory_map()
    for gpu_idx, memory_used in gpu_memory_map.items():
        if memory_used <= max_memory and gpu_idx not in unallowed_gpus:
            logger.debug(f'Using GPU {gpu_idx}')
            return gpu_idx
    logger.debug('No allowed free GPUs')
    return None


def to_torch_embedding(embedding_matrix: np.ndarray) -> torch.nn.Embedding:
    """Transform a numpy matrix into a torch Embedding object"""

    torch_embedding = torch.nn.Embedding(*embedding_matrix.shape)
    torch_embedding.weight = torch.nn.Parameter(torch.Tensor(embedding_matrix))

    return torch_embedding
