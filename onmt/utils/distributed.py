""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""


from __future__ import print_function

import math
import pickle

import torch.distributed

from onmt.utils.logging import logger


def is_master(device_id):
    return device_id == 0


def multi_init(opt, device_id):
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=opt.master_ip, master_port=opt.master_port
    )
    dist_world_size = opt.world_size
    torch.distributed.init_process_group(
        backend=opt.gpu_backend,
        init_method=dist_init_method,
        world_size=dist_world_size,
        rank=device_id,
    )
    gpu_rank = torch.distributed.get_rank()
    if not is_master(device_id):
        logger.disabled = False

    return gpu_rank


def all_reduce_and_rescale_tensors(
    tensors, rescale_denom, buffer_size=10485760, group=None
):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes
        group: communication group
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = (
        tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    )
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for tensor in buffer:
            num_el = tensor.numel()
            buffer_t[offset: offset + num_el].copy_(tensor.view(-1))
            offset += num_el

        # all-reduce and rescale
        if group:
            torch.distributed.all_reduce(buffer_t[:offset], group=group)
        else:
            torch.distributed.all_reduce(
                buffer_t[:offset]
            )  # no idea why doesn't like group=None
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for tensor in buffer:
            num_el = tensor.numel()
            tensor.view(-1).copy_(buffer_t[offset: offset + num_el])
            offset += num_el

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            if group:
                torch.distributed.all_reduce(t, group=group)
            else:
                torch.distributed.all_reduce(t)  # no idea why doesn't like group=None
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_reduce_tensors_init(tensors, group=None):
    for t in tensors:
        if group == None:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        else:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if (
        not hasattr(all_gather_list, "_in_buffer")
        or max_size != all_gather_list._in_buffer.size()
    ):
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size) for _ in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError("encoded data exceeds max_size: {}".format(enc_size + 2))
    assert max_size < 255 * 256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2: enc_size + 2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2: size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results


class CommunicationGroup:
    def __init__(
        self,
        torch_dist_group: torch.distributed.distributed_c10d.ProcessGroupNCCL,
        group_size: int,
    ):
        self.torch_dist_group = torch_dist_group
        self.size = group_size
