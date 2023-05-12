import sten
import torch
import math
from pathlib import Path
import sys
import ctypes

from .grouped_nm_tensor import GroupedNMTensor
from . import matmul_generator
from .dace_gnm_mult import nmg_mult


class GroupedNMSparsifier:
    def __init__(self, n, m, g):
        self.n = n
        self.m = m
        self.g = g


@sten.register_sparsifier_implementation(
    sparsifier=GroupedNMSparsifier, inp=torch.Tensor, out=GroupedNMTensor
)
def dense_to_grouped_nm(sparsifier, tensor, grad_fmt=None):
    gnm = GroupedNMTensor.from_dense(
        tensor,
        sparsifier.n,
        sparsifier.m,
        sparse_dim=tensor.ndim - 2,
        group_size=sparsifier.g,
        group_dim=tensor.ndim - 1,
    )
    res = sten.SparseTensorWrapper.wrapped_from_dense(
        gnm,
        tensor,
        grad_fmt,
    )
    return res


LOADED_LIBS = {}  # Library keepalive
LOADED_FUNCS = {}  # Preloaded functions


def sparse_dense_mul_dispatch(
    nm_strides, DM, DK, sparse_values, sparse_indices, dense, trans_a, trans_b, trans_c
):
    assert len(dense.shape) == 2
    if trans_a:
        DM, DK = DK, DM
    DN = dense.shape[0 if trans_b else 1]

    # Set transposed B value based on contiguity (if b.T is contiguous, trans_b=True)
    trans_b = not dense.is_contiguous()

    kernel = nm_strides["kernel"]

    DK3 = nm_strides["group_size"]
    DM3S = nm_strides["n"]  # this is also first dim of accumulator size
    DM3 = nm_strides["m"]

    DM2 = nm_strides["tile_a"]  # cache tile size of A
    DN4 = nm_strides["acc_width"]  # second dim of accumulator size
    DN2 = nm_strides["tile_b"]  # cache tile size of B
    DN3 = 1  # redundant tile dimension, unused at the momemnt (always 1)
    DN5 = 8 if kernel == "avx2" else 16  # vector size (in floats)

    DK2 = math.factorial(DM3) // math.factorial(DM3S) // math.factorial(DM3 - DM3S)

    params = matmul_generator.derive_params(
        {
            "DK3": DK3,
            "DM3": DM3,
            "DM3S": DM3S,
            "DM2": DM2,
            "DN5": DN5,
            "DN4": DN4,
            "DN3": DN3,
            "DN2": DN2,
            "DM": DM,
            "DK": DK,
            "DN": DN,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "trans_c": trans_c,
        }
    )

    encoded_name = "nmg_" + matmul_generator.params_hash(params)
    if encoded_name in LOADED_FUNCS:
        sparse_dense_impl = LOADED_FUNCS[encoded_name]
    else:
        path = f".dacecache/{encoded_name}/build/lib{encoded_name}.so"
        if not Path(path).is_file():
            print("Compilation started...", file=sys.stderr)
            nmg_mult(
                (DM, DK, DN),
                m=DM3,
                n=DM3S,
                g=DK3,
                transpose_b=trans_b,
                transpose_c=trans_c,
                kernel=kernel,
                tile=0,
                tile_2=0,
                local_b=True,
                local_c=True,
                name=encoded_name,
            )
            print("Compilation completed", file=sys.stderr)

        lib = ctypes.CDLL(path)
        sparse_dense_impl = getattr(lib, f"__program_{encoded_name}")
        sparse_dense_impl.argtypes = [
            ctypes.c_void_p,  # void* state = NULL,
            ctypes.c_void_p,  # int16_t* __restrict__ A_idx,
            ctypes.c_void_p,  # float* __restrict__ A_val,
            ctypes.c_void_p,  # float* __restrict__ B,
            ctypes.c_void_p,  # float* __restrict__ C,
            ctypes.c_void_p,  # int* __restrict__ groups  = NULL
        ]

        LOADED_LIBS[encoded_name] = lib
        LOADED_FUNCS[encoded_name] = sparse_dense_impl

    DM_padded = math.ceil(DM / DM3) * DM3

    output = torch.empty(DN, DM_padded) if trans_c else torch.empty(DM_padded, DN)
    svc = sparse_values.contiguous()
    sic = sparse_indices.contiguous()

    sparse_dense_impl(
        None,
        sic.data_ptr(),
        svc.data_ptr(),
        dense.data_ptr(),
        output.data_ptr(),
        None,
    )

    return output


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, GroupedNMTensor, torch.Tensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def sparse_torch_nn_functional_linear(ctx, inputs, output_sparsifiers):
    input, weight, bias = inputs
    ctx.save_for_backward(input, weight, bias)

    flattened_input = torch.flatten(input, start_dim=0, end_dim=-2)

    sparse_values = weight.wrapped_tensor.val
    sparse_indices = weight.wrapped_tensor.idx
    DM, DK = weight.wrapped_tensor.nm_strides["dense_shape"]

    output = sparse_dense_mul_dispatch(
        weight.wrapped_tensor.nm_strides,
        DM,
        DK,
        sparse_values,
        sparse_indices,
        flattened_input,
        trans_a=False,
        trans_b=True,
        trans_c=True,
    )  # this is supposed to be more efficient, but unfortunately it is slower

    output = output.reshape((*input.shape[0:-1], -1))[..., :DM]

    if bias is not None:
        output += bias.unsqueeze(0).expand_as(output)
    return output
