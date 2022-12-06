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

if torch.__version__.startswith("1.12"):
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
