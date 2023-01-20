import torch

import inspect
import sys
import copy
from functools import partial, cache

from sten import (
    SparseTensorWrapper,
    flatten_list_of_tensors_in_args,
    unflatten_list_of_tensors_in_args,
)
import sten
import warnings

from sten import DispatchError
from sten import make_sparse_catcher


@cache
def get_member_funcs(subclass):
    return [
        (n, m)
        for (n, m) in inspect.getmembers(subclass)
        if (not inspect.isclass(m) and callable(m))
    ]


@cache
def get_mod_classes(module):
    return inspect.getmembers(module, inspect.isclass)


def patch_class(old, new, subclass):
    for name, func in get_member_funcs(subclass):
        if func is old:
            setattr(subclass, name, new)


def patch_module(old, new, module):
    for name, subclass in get_mod_classes(module):
        patch_class(old, new, subclass)
    for name, func in get_member_funcs(module):
        if func is old:
            setattr(module, name, new)


TORCH_MODULES = [mod for name, mod in sys.modules.items() if name.startswith("torch")]


def patch(old, new, module_scope=TORCH_MODULES):
    for mod in module_scope:
        patch_module(old, new, mod)


# +++++ heuristically try to catch most of pytorch API calls

# for nc, c in inspect.getmembers(torch._C._distributed_c10d, inspect.isclass):
#     for nm, m in inspect.getmembers(c):
#         if callable(m) and not nm.startswith('__'):
#             warnings.warn(f"Patching {nc}.{nm}")
#             assert type(m) is not property
#             patch(m, make_sparse_catcher(m))

# for nf, f in inspect.getmembers(torch._C._distributed_c10d, inspect.isbuiltin):
#     warnings.warn(f"Patching {nf}")
#     patch(f, make_sparse_catcher(f))

# for nf, f in inspect.getmembers(torch._C):
#     if inspect.isbuiltin(f) and nf.startswith('_'):
#         patch(f, partial(sparse_catcher, f))

# ===== heuristically try to catch most of pytorch API calls

# fixes torch DDP
for x in [
    torch._C._distributed_c10d._verify_params_across_processes,
    torch._C._distributed_c10d._broadcast_coalesced,
    torch._C._distributed_c10d._compute_bucket_assignment_by_size,
]:
    patch(x, make_sparse_catcher(x))


tva = int(torch.__version__.split(".")[0])
tvb = int(torch.__version__.split(".")[1])
if int(tva * 100 + tvb) > 112:
    # patch transformer code to remove control flow and make it traceable
    def patched_forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    patch(torch.nn.TransformerEncoderLayer.forward, patched_forward)


# patch Nvidia Apex (https://github.com/NVIDIA/apex)
try:
    import apex
    import amp_C

    patch_scope = [
        mod for name, mod in sys.modules.items() if name.startswith(("apex", "amp_C"))
    ]
    patch(
        amp_C.multi_tensor_l2norm,
        sten.make_sparse_catcher(amp_C.multi_tensor_l2norm),
        patch_scope,
    )
    patch(
        amp_C.multi_tensor_lamb,
        sten.make_sparse_catcher(amp_C.multi_tensor_lamb),
        patch_scope,
    )
except ImportError:
    pass


# patch PyTorch Distributed Data Parallel


def sparse_ddp_all_reduce_hook(state, bucket):
    dense_buf = bucket.buffer()

    total_elems = 0
    for p in bucket.parameters():
        if isinstance(p, sten.SparseTensorWrapper):
            total_elems += p.numel()

    # reduce all sparse tensors
    sparse_buf = torch.zeros(
        max(total_elems, 1), dtype=dense_buf.dtype, device=dense_buf.device
    )
    processed = 0
    for p in bucket.parameters():
        if isinstance(p, sten.SparseTensorWrapper):
            sparse_buf[
                processed : processed + p.numel()
            ] = p.grad.wrapped_tensor.to_dense().flatten()
            processed += p.numel()
    assert processed == total_elems

    sparse_buf /= torch.distributed.get_world_size()
    fut_sparse = torch.distributed.all_reduce(
        sparse_buf, op=torch.distributed.ReduceOp.SUM, async_op=True
    ).get_future()

    fut_sparse.wait()

    # reduce all dense tensors
    dense_buf /= torch.distributed.get_world_size()
    fut_dense = torch.distributed.all_reduce(
        dense_buf, op=torch.distributed.ReduceOp.SUM, async_op=True
    ).get_future()

    fut_dense.wait()

    fut = torch.futures.collect_all([fut_sparse, fut_dense])

    def postporcess(fut):
        spase_fut, dense_fut = fut.value()
        [sparse_buf] = spase_fut.value()
        [dense_buf] = dense_fut.value()
        # process sparse gradients maqnually
        processed = 0
        for p in bucket.parameters():
            if isinstance(p, sten.SparseTensorWrapper):
                dense_grad = sparse_buf[processed : processed + p.numel()].reshape(
                    p.shape
                )
                sparsifier = sten.get_sparsifier_implementation(
                    sten.SameFormatSparsifier,
                    torch.Tensor,
                    p.grad.wrapped_tensor.__class__,
                )
                reduced_sparse_grad = sparsifier(
                    sten.SameFormatSparsifier(p.grad), dense_grad
                )
                p.grad.init_from_other(reduced_sparse_grad)
                processed += p.numel()
        assert processed == total_elems
        # return dense_buf as is, it will be used to update grad values of dense tensors by DDP
        return dense_buf

    return fut.then(postporcess)


def patch_ddp():
    orig_ddp_init = torch.nn.parallel.DistributedDataParallel.__init__
    patch_scope = [
        mod for name, mod in sys.modules.items() if name.startswith(("torch"))
    ]

    def my_ddp_init(*args, **kwargs):
        orig_ddp_init(*args, **kwargs)
        self, module, *_ = args
        # add comm hook only if model has sparse parameters
        if any(isinstance(p, sten.SparseParameterWrapper) for p in module.parameters()):
            self.register_comm_hook(state=None, hook=sparse_ddp_all_reduce_hook)

    patch(orig_ddp_init, my_ddp_init, patch_scope)


patch_ddp()
