import torch

import logging
import inspect
import sys
import copy
from functools import partial, cache

from sten.sten import SparseTensorWrapper, flatten_list_of_tensors_in_args, unflatten_list_of_tensors_in_args
import sten

_log = logging.getLogger(__name__)

@cache
def get_member_funcs(subclass):
    return [(n, m) for (n, m) in inspect.getmembers(subclass) if (not inspect.isclass(m) and callable(m))]

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

TORCH_MODULES = {name: mod for name, mod in sys.modules.items() if name.startswith('torch')}



def patch(old, new, module_scope=None):
    for mod in TORCH_MODULES.values():
        patch_module(old, new, mod)
        

def make_sparse_catcher(orig_fn):
    def sparse_catcher(*args, **kwargs):
        args_with_stubs, flat_args = flatten_list_of_tensors_in_args(args)
        kwargs_with_stubs, flat_kwargs = flatten_list_of_tensors_in_args(kwargs)
        if flat_args or flat_kwargs:
            # implementation that will handle args properly
            #_log.warning(f"Catching {orig_fn.__module__}.{orig_fn.__name__} called with the sparse arguments!")
            flat_d_args = sten.densify(flat_args)
            flat_d_kwargs = sten.densify(flat_kwargs)
            d_args = unflatten_list_of_tensors_in_args(args_with_stubs, flat_d_args)
            d_kwargs = unflatten_list_of_tensors_in_args(kwargs_with_stubs, flat_d_kwargs)
            arg_copies = [(copy.deepcopy(dense_ten) if isinstance(orig_ten, sten.SparseTensorWrapper) else None) for orig_ten, dense_ten in zip(flat_args + flat_kwargs, flat_d_args + flat_d_kwargs)]
            d_output = orig_fn(*d_args, **d_kwargs)
            out_with_stabs, flat_out = flatten_list_of_tensors_in_args(d_output)
            # check for modifications
            for cpy, orig, dense in zip(arg_copies, flat_args + flat_kwargs, flat_d_args + flat_d_kwargs):
                if isinstance(orig, SparseTensorWrapper):
                    if torch.allclose(cpy, dense):
                        continue  # no inplace changes
                    sparsifier = sten.get_sparsifier_implementation(
                        sten.SameFormatSparsifier, torch.Tensor, orig.wrapped_tensor.__class__
                    )
                    # TODO: not sure how to distinguish full replacement and nonzero modification
                    sparse_arg = sparsifier(sten.SameFormatSparsifier(orig), dense)
                    orig.init_from_other(sparse_arg)
                else:
                    assert cpy is None
            # return output
            if flat_out:
                # TODO: need to keep track of input-output tensor references in flat lists
                raise NotImplementedError()
            return d_output
        else:
            # default implementation
            return orig_fn(*args, **kwargs)
    return sparse_catcher

# +++++ heuristically try to catch most of pytorch API calls 

for nc, c in inspect.getmembers(torch._C._distributed_c10d, inspect.isclass):
    for nm, m in inspect.getmembers(c):
        if callable(m) and not nm.startswith('__'):
            _log.warning(f"Patching {nc}.{nm}")
            assert type(m) is not property
            patch(m, make_sparse_catcher(m))
        
for nf, f in inspect.getmembers(torch._C._distributed_c10d, inspect.isbuiltin):
    _log.warning(f"Patching {nf}")
    patch(f, make_sparse_catcher(f))
        
# for nf, f in inspect.getmembers(torch._C):
#     if inspect.isbuiltin(f) and nf.startswith('_'):
#         patch(f, partial(sparse_catcher, f))

# ===== heuristically try to catch most of pytorch API calls 


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
