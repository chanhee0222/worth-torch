# From fairseq.distributed_utils
# to avoid circular imports

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import torch
import torch.distributed as dist


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_default_group():
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256

    cpu_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    cpu_buffer[1] = enc_size % 255
    cpu_buffer[2 : enc_size + 2] = torch.ByteTensor(list(enc))
    start = rank * max_size
    size = enc_size + 2
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            size = (255 * item(out_buffer[0])) + item(out_buffer[1])
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[2 : size + 2].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor