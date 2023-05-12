import torch
from math import factorial as fact
import math
import itertools
from native_scripting import compile
import ctypes
import numpy as np
import copy
import functools
import time
from sympy import factorint
from typing import Tuple
import warnings
import cpuinfo

try:
    cache = functools.cache
except AttributeError:
    cache = functools.lru_cache(maxsize=None)

# ++++++++++++++ n:m order generator ++++++++++++++


def is_valid_n_m(l, n, m):
    def is_increasing(a):
        return all([e1 < e2 for e1, e2 in zip(a[:-1], a[1:])])

    def is_adjacent(a1, a2):
        return sum([i1 != i2 for i1, i2 in zip(a1, a2)]) == 1

    all_increasing = all([is_increasing(a) for a in l])
    all_adjacent = all([is_adjacent(a, b) for a, b in zip(l[:-1], l[1:])])
    correct_size = len(l) == fact(m) // fact(n) // fact(m - n)
    return all_increasing and all_adjacent and correct_size


@cache
def make_n_m(n, m, special=False):
    # special -- True: "..xx.->...xx" False: "xx...->...xx"
    assert 0 <= n and n <= m
    if n == m or n == 0:
        return [tuple(range(n))]
    first = make_n_m(n, m - 1, True)
    if special:
        first = list(reversed(first))
    second = make_n_m(n - 1, m - 1, not special)
    second = [tpl + (m - 1,) for tpl in second]
    result = first + second
    assert is_valid_n_m(result, n, m)
    return result


def make_n_m_mask(m, nnz_indices):
    res = [0] * m
    for idx in nnz_indices:
        res[idx] = 1
    return tuple(res)


def make_n_m_order_c(nnz_indices_list):
    elems = []
    for tpl in nnz_indices_list:
        elem = "{" + ", ".join([str(t) for t in tpl]) + "}"
        elems.append(elem)
    return "{" + ", ".join(elems) + "}"


# ============== n:m order generator ==============


@cache
def compute_nm_strides(dense_shape, n, m, sparse_dim, group_dim, group_size):
    chunk_size = fact(m) // fact(n) // fact(m - n)
    num_chunks = math.ceil(dense_shape[group_dim] / (chunk_size * group_size))
    num_blocks = math.ceil(dense_shape[sparse_dim] / m)
    sparse_dim_expanded = [num_blocks, m]
    group_dim_expanded = [num_chunks, chunk_size, group_size]

    padded_sparse_dim = math.prod(sparse_dim_expanded)
    padded_group_dim = math.prod(group_dim_expanded)

    # pad sparse dim to fit into n:m blocks
    padded_dense_shape = list(dense_shape)
    padded_dense_shape[sparse_dim] = padded_sparse_dim
    padded_dense_shape[group_dim] = padded_group_dim

    if sparse_dim < group_dim:
        exp_sparse_dim = sparse_dim
        exp_group_dim = group_dim + len(sparse_dim_expanded) - 1

        smaller_dim_from = exp_sparse_dim
        smaller_dim_to = exp_sparse_dim + len(sparse_dim_expanded)
        larger_dim_from = exp_group_dim
        larger_dim_to = exp_group_dim + len(group_dim_expanded)

        expanded_dense_shape = (
            padded_dense_shape[:sparse_dim]
            + sparse_dim_expanded
            + padded_dense_shape[sparse_dim + 1 : group_dim]
            + group_dim_expanded
            + padded_dense_shape[group_dim + 1 :]
        )

    else:
        exp_sparse_dim = sparse_dim + len(group_dim_expanded) - 1
        exp_group_dim = group_dim

        smaller_dim_from = exp_group_dim
        smaller_dim_to = exp_group_dim + len(group_dim_expanded)
        larger_dim_from = exp_sparse_dim
        larger_dim_to = exp_sparse_dim + len(sparse_dim_expanded)

        expanded_dense_shape = (
            padded_dense_shape[:group_dim]
            + group_dim_expanded
            + padded_dense_shape[group_dim + 1 : sparse_dim]
            + sparse_dim_expanded
            + padded_dense_shape[sparse_dim + 1 :]
        )

    sparse_val_shape = (
        expanded_dense_shape[: exp_sparse_dim + 1]
        + [n]
        + expanded_dense_shape[exp_sparse_dim + 2 :]
    )
    sparse_idx_shape = (
        expanded_dense_shape[: exp_sparse_dim + 1]
        + [1]
        + expanded_dense_shape[exp_sparse_dim + 2 :]
    )

    if sparse_dim < group_dim:
        dense_block_stride = math.prod(expanded_dense_shape[smaller_dim_to:])
        dense_group_stride = math.prod(expanded_dense_shape[larger_dim_to:])
        sparse_block_stride = math.prod(sparse_val_shape[smaller_dim_to:])
        sparse_group_stride = math.prod(sparse_val_shape[larger_dim_to:])
        idx_block_stride = math.prod(sparse_idx_shape[smaller_dim_to:])
        idx_group_stride = math.prod(sparse_idx_shape[larger_dim_to:])
    else:
        dense_block_stride = math.prod(expanded_dense_shape[larger_dim_to:])
        dense_group_stride = math.prod(expanded_dense_shape[smaller_dim_to:])
        sparse_block_stride = math.prod(sparse_val_shape[larger_dim_to:])
        sparse_group_stride = math.prod(sparse_val_shape[smaller_dim_to:])
        idx_block_stride = math.prod(sparse_idx_shape[larger_dim_to:])
        idx_group_stride = math.prod(sparse_idx_shape[smaller_dim_to:])

    dense_loop_outer_stride = math.prod(expanded_dense_shape[smaller_dim_from + 1 :])
    dense_loop_middle_stride = math.prod(expanded_dense_shape[larger_dim_from + 1 :])
    sparse_loop_outer_stride = math.prod(sparse_val_shape[smaller_dim_from + 1 :])
    sparse_loop_middle_stride = math.prod(sparse_val_shape[larger_dim_from + 1 :])
    idx_loop_outer_stride = math.prod(sparse_idx_shape[smaller_dim_from + 1 :])
    idx_loop_middle_stride = math.prod(sparse_idx_shape[larger_dim_from + 1 :])

    expanded_ndim = len(expanded_dense_shape)

    last_dims = [
        exp_group_dim,
        exp_sparse_dim,
        exp_group_dim + 1,
        exp_group_dim + 2,
        exp_sparse_dim + 1,
    ]
    dim_permutation = [
        i for i in range(expanded_ndim) if i not in last_dims
    ] + last_dims
    inverse_dim_permutation = sorted(
        range(expanded_ndim), key=lambda x: dim_permutation[x]
    )

    permuted_shape = [expanded_dense_shape[i] for i in dim_permutation]
    merged_shape = [math.prod(permuted_shape[:-3]), *permuted_shape[-3:]]

    return {
        "n": n,
        "m": m,
        "sparse_dim": sparse_dim,
        "group_dim": group_dim,
        "group_size": group_size,
        "dense_shape": dense_shape,
        "padded_dense_shape": padded_dense_shape,
        "expanded_dense_shape": expanded_dense_shape,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "num_blocks": num_blocks,
        "exp_sparse_dim": exp_sparse_dim,
        "exp_group_dim": exp_group_dim,
        "smaller_dim_from": smaller_dim_from,
        "smaller_dim_to": smaller_dim_to,
        "larger_dim_from": larger_dim_from,
        "larger_dim_to": larger_dim_to,
        "sparse_val_shape": sparse_val_shape,
        "sparse_idx_shape": sparse_idx_shape,
        "loop_outer_size": math.prod(expanded_dense_shape[: smaller_dim_from + 1]),
        "loop_middle_size": math.prod(
            expanded_dense_shape[smaller_dim_to : larger_dim_from + 1]
        ),
        "loop_inner_size": math.prod(expanded_dense_shape[larger_dim_to:]),
        "order": make_n_m(n, m),
        "dense_block_stride": dense_block_stride,
        "dense_group_stride": dense_group_stride,
        "sparse_block_stride": sparse_block_stride,
        "sparse_group_stride": sparse_group_stride,
        "idx_block_stride": idx_block_stride,
        "idx_group_stride": idx_group_stride,
        "dense_loop_middle_stride": dense_loop_middle_stride,
        "dense_loop_outer_stride": dense_loop_outer_stride,
        "sparse_loop_middle_stride": sparse_loop_middle_stride,
        "sparse_loop_outer_stride": sparse_loop_outer_stride,
        "idx_loop_middle_stride": idx_loop_middle_stride,
        "idx_loop_outer_stride": idx_loop_outer_stride,
        "padded_sparse_dim": padded_sparse_dim,
        "padded_group_dim": padded_group_dim,
        "dim_permutation": dim_permutation,
        "inverse_dim_permutation": inverse_dim_permutation,
        "permuted_shape": permuted_shape,
        "merged_shape": merged_shape,
    }


@cache
def get_dense_to_grouped_nm_impl_cpu(dense_dtype, dense_shape, n):
    assert len(dense_shape) == 4  # (batch_dim, chunk=m!/n!/(m-n)!, group, block=m)
    chunk_size = dense_shape[1]
    group_size = dense_shape[2]
    m = dense_shape[3]
    assert chunk_size == fact(m) // fact(n) // fact(m - n)

    # check that group_size * chunk_size fits int16_t
    assert chunk_size * group_size < 2**15

    order = make_n_m(n, m)
    loop_outer_size = dense_shape[0]

    dense_group_stride = m
    sparse_group_stride = n
    idx_group_stride = 1

    dense_loop_outer_stride = chunk_size * group_size * m
    sparse_loop_outer_stride = chunk_size * group_size * n
    idx_loop_outer_stride = chunk_size * group_size * 1

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        static const int8_t nnz_order[{chunk_size}][{n}] = {make_n_m_order_c(order)};
        extern "C" void func({dtype}* dense, {dtype}* sparse, int16_t* idx) {{
            const int64_t acc_size = {chunk_size} * {chunk_size} * {group_size};
            std::vector<std::tuple<{dtype}, int16_t, int16_t>> accs(acc_size);
            for (int64_t os_idx = 0; os_idx < {loop_outer_size}; os_idx++) {{
                {dtype}* dense_base = &dense[os_idx * {dense_loop_outer_stride}];
                {dtype}* sparse_base = &sparse[os_idx * {sparse_loop_outer_stride}];
                int16_t* idx_base = &idx[os_idx * {idx_loop_outer_stride}];
                for (int64_t cs_idx = 0; cs_idx < {chunk_size}; cs_idx++) {{
                    for (int64_t gs_idx = 0; gs_idx < {group_size}; gs_idx++) {{
                        for (int64_t nnz_idx = 0; nnz_idx < {chunk_size}; nnz_idx++) {{
                            {dtype} blk_acc = 0;
                            int16_t original_idx = cs_idx * {group_size} + gs_idx;
                            for (int64_t n_idx = 0; n_idx < {n}; n_idx++) {{
                                int16_t m_idx = nnz_order[nnz_idx][n_idx];
                                blk_acc += std::abs(dense_base[m_idx + original_idx * {dense_group_stride}]);
                            }}
                            int64_t acc_idx = nnz_idx + original_idx * {chunk_size};
                            accs[acc_idx] = std::make_tuple(blk_acc, original_idx, nnz_idx);
                        }}
                    }}
                }}
                std::sort(accs.begin(), accs.end(), std::greater<>());
                // extract elements from sorted accumulators into resulting tensor
                int16_t group_elems[{chunk_size}] = {{ 0 }};
                bool is_taken[{chunk_size} * {group_size}] = {{ 0 }};
                for (int64_t acc_idx = 0; acc_idx < acc_size; acc_idx++) {{
                    {dtype} val = 0;
                    int16_t orig_idx = -1, nnz_idx = -1;
                    std::tie(val, orig_idx, nnz_idx) = accs[acc_idx];
                    if (!is_taken[orig_idx] && group_elems[nnz_idx] < {group_size}) {{
                        is_taken[orig_idx] = true;
                        int16_t group_idx = group_elems[nnz_idx]++;
                        int16_t sparse_idx = (group_idx + nnz_idx * {group_size});
                        // put into the output
                        idx_base[sparse_idx * {idx_group_stride}] = orig_idx;
                        for (int64_t n_idx = 0; n_idx < {n}; n_idx++) {{
                            int64_t m_idx = nnz_order[nnz_idx][n_idx];
                            sparse_base[n_idx + sparse_idx * {sparse_group_stride}] = dense_base[m_idx + orig_idx * {dense_group_stride}];
                        }}
                    }}
                }}
            }}
        }}
        """,
    )
    lib.func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func


def int_ceil(x, y):
    return (x - 1) // y + 1


def find_opt_cuda_block(nx, ny, nz, max_block=1024, max_z=64):
    # prioritize powers of 2 to avoid remainders everywhere
    dims = [nx, ny, nz]
    pows2 = [factorint(d).get(2, 0) for d in dims]
    block_sizes = [2**p for p in pows2]
    dims = [d // b for d, b in zip(dims, block_sizes)]

    # fit z dimension if it exceedes limits
    while block_sizes[-1] > max_z:
        block_sizes[-1] //= 2
        dims[-1] *= 2

    # remove some powers if max block size is exceeded
    while math.prod(block_sizes) > max_block:
        for i in range(len(dims)):
            if block_sizes[i] == 1:
                continue
            block_sizes[i] //= 2
            dims[i] *= 2
            break

    assert math.prod(block_sizes) <= max_block
    # now distribute some powers of 2 while max block is not reached
    while math.prod(block_sizes) < max_block:
        # print(f'block sizes {block_sizes}')
        exitting = False
        for i, d in enumerate(dims):
            if d != max(dims):
                continue
            if i == 2 and block_sizes[i] == 64:
                exitting = True
                break
            block_sizes[i] *= 2
            dims[i] //= 2
            break
        if exitting:
            break

    # print(f"optimal block {block_sizes} for inputs of size {[nx, ny, nz]}")
    assert math.prod(block_sizes) <= max_block
    assert block_sizes[-1] <= max_z
    return block_sizes


@cache
def get_dense_to_grouped_nm_impl_cuda(dense_dtype, dense_shape, n):
    assert len(dense_shape) == 4  # (batch_dim, chunk=m!/n!/(m-n)!, group, block=m)
    chunk_size = dense_shape[1]
    group_size = dense_shape[2]
    m = dense_shape[3]
    assert chunk_size == fact(m) // fact(n) // fact(m - n)

    # check that group_size * chunk_size fits int16_t
    assert chunk_size * group_size < 2**15

    order = make_n_m(n, m)
    loop_outer_size = dense_shape[0]

    dense_group_stride = m
    sparse_group_stride = n
    idx_group_stride = 1

    dense_loop_outer_stride = chunk_size * group_size * m
    sparse_loop_outer_stride = chunk_size * group_size * n
    idx_loop_outer_stride = chunk_size * group_size * 1

    acc_size = chunk_size * chunk_size * group_size

    block_x, block_y, block_z = find_opt_cuda_block(
        chunk_size, loop_outer_size, group_size
    )

    grid_x = int_ceil(chunk_size, block_x)
    grid_y = int_ceil(loop_outer_size, block_y)
    grid_z = int_ceil(group_size, block_z)

    N = chunk_size
    if N % 2 == 0:
        Nx = N - 1
        Ny = N // 2
    else:
        Nx = N
        Ny = (N - 1) // 2

    block_x_pp, block_y_pp, block_z_pp = find_opt_cuda_block(
        Nx * group_size, loop_outer_size, Ny * group_size
    )

    grid_x_pp = int_ceil(Nx * group_size, block_x_pp)
    grid_y_pp = int_ceil(loop_outer_size, block_y_pp)
    grid_z_pp = int_ceil(Ny * group_size, block_z_pp)

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <iostream>
        #define CUDA_CHECK(expr) do {{\\
            cudaError_t err = (expr);\\
            if (err != cudaSuccess) {{\\
                std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\\n"; \\
                abort(); \\
            }}\\
        }} while(0)
        __device__ __managed__ int something_swapped;
        __device__ static const int8_t nnz_order[{chunk_size}][{n}] = {make_n_m_order_c(order)};
        __global__ void kernel_preprocess({dtype}* dense, {dtype}* all_accs, int* is_taken_all, int* groups_all) {{
            int64_t cs_idx = threadIdx.x + blockDim.x * blockIdx.x;
            int64_t os_idx = threadIdx.y + blockDim.y * blockIdx.y;
            int64_t gs_idx = threadIdx.z + blockDim.z * blockIdx.z;
            if (cs_idx >= {chunk_size}) return;
            if (os_idx >= {loop_outer_size}) return;
            if (gs_idx >= {group_size}) return;
            
            int* groups = groups_all + os_idx * {chunk_size};
            
            {dtype}* accs = all_accs + os_idx * {acc_size};
            {dtype}* dense_base = &dense[os_idx * {dense_loop_outer_stride}];
            int max_nnz_idx = -1;
            {dtype} max_val = 1e-3;
            for (int16_t nnz_idx = 0; nnz_idx < {chunk_size}; nnz_idx++) {{
                {dtype} blk_acc = 0;
                int16_t original_idx = cs_idx * {group_size} + gs_idx;
                for (int64_t n_idx = 0; n_idx < {n}; n_idx++) {{
                    int16_t m_idx = nnz_order[nnz_idx][n_idx];
                    {dtype} abs_val = std::abs(dense_base[m_idx + original_idx * {dense_group_stride}]);
                    blk_acc += abs_val;
                }}
                int64_t acc_idx = nnz_idx + original_idx * {chunk_size};
                accs[acc_idx] = blk_acc;
                if (blk_acc > max_val) {{
                    max_nnz_idx = nnz_idx;
                    max_val = blk_acc;
                }}
            }}
            int group_idx = {group_size};
            if (max_nnz_idx != -1) {{
                group_idx = atomicAdd(groups + max_nnz_idx, 1);
            }}
            // is_taken maps index in the original array to the index in new array
            int* is_taken = is_taken_all + os_idx * {chunk_size} * {group_size};
            if (group_idx < {group_size}) {{
                is_taken[cs_idx * {group_size} + gs_idx] = max_nnz_idx * {group_size} + group_idx;
            }} else {{
                something_swapped = 1; //  we didn't find optimal distribution on the first try
                is_taken[cs_idx * {group_size} + gs_idx] = -1;
            }}
        }}
        __device__ void vset(volatile int* ptr, int val) {{ *ptr = val; }} 
        __device__ int vget(volatile int* ptr) {{ return *ptr; }} 
        __global__ void kernel_preprocess2(int* is_taken_all, int* groups_all) {{
            int64_t cs_idx = threadIdx.x + blockDim.x * blockIdx.x;
            int64_t os_idx = threadIdx.y + blockDim.y * blockIdx.y;
            int64_t gs_idx = threadIdx.z + blockDim.z * blockIdx.z;
            if (cs_idx >= {chunk_size}) return;
            if (os_idx >= {loop_outer_size}) return;
            if (gs_idx >= {group_size}) return;
            int* is_taken = is_taken_all + os_idx * {chunk_size} * {group_size};
            int* groups = groups_all + os_idx * {chunk_size};
            if (is_taken[cs_idx * {group_size} + gs_idx] == -1) {{
                for (int16_t nnz_idx = 0; nnz_idx < {chunk_size}; nnz_idx++) {{
                    if (vget(groups + nnz_idx) < {group_size}) {{
                        int group_idx = atomicAdd(groups + nnz_idx, 1);
                        if (group_idx < {group_size}) {{
                            is_taken[cs_idx * {group_size} + gs_idx] = nnz_idx * {group_size} + group_idx;
                            break;
                        }}
                    }}
                }}
            }}
        }}
        __global__ void kernel_postprocess({dtype}* all_accs, int* is_taken_all) {{            
            int64_t xgs_idx = threadIdx.x + blockDim.x * blockIdx.x;
            int64_t os_idx = threadIdx.y + blockDim.y * blockIdx.y;
            int64_t ygs_idx = threadIdx.z + blockDim.z * blockIdx.z;
            if (xgs_idx >= {Nx} * {group_size}) return;
            if (os_idx >= {loop_outer_size}) return;
            if (ygs_idx >= {Ny} * {group_size}) return;
            
            {dtype}* accs = all_accs + os_idx * {acc_size};
            // extract elements from sorted accumulators into resulting tensor
            int* is_taken = is_taken_all + os_idx * {chunk_size} * {group_size};
            
            int x = xgs_idx / {group_size};
            int l_old_gs_idx = xgs_idx % {group_size};
            
            int y = ygs_idx / {group_size};
            int r_old_gs_idx = ygs_idx % {group_size};
            
            int l_old_cs_idx = -1;
            int r_old_cs_idx = -1;
            
            // init l_old_cs_idx and r_old_cs_idx
            if ({N} % 2 == 0) {{
                if (y > x) {{
                    l_old_cs_idx = ({N} - 1) - x;
                    r_old_cs_idx = ({N} - 1) - y;
                }} else {{
                    l_old_cs_idx = x + 1;
                    r_old_cs_idx = y;
                }}
            }} else {{                
                if (y >= x) {{
                    l_old_cs_idx = ({N} - 1) - x;
                    r_old_cs_idx = ({N} - 2) - y;
                }} else {{
                    l_old_cs_idx = x;
                    r_old_cs_idx = y;
                }}
            }}
            
            for (int rep = 0; rep < 1; rep++) {{
            
                int l_old_idx = l_old_cs_idx * {group_size} + l_old_gs_idx;
                int r_old_idx = r_old_cs_idx * {group_size} + r_old_gs_idx;
                
                if (l_old_idx > r_old_idx) {{
                    // guarantee locking in the uniform order
                    int tmp = l_old_idx;
                    l_old_idx = r_old_idx;
                    r_old_idx = tmp;
                }}
                
                int* l_is_taken = is_taken + l_old_idx;
                int* r_is_taken = is_taken + r_old_idx;
                
                int l_new_idx = vget(l_is_taken);
                if (l_new_idx < 0) continue;
                int r_new_idx = vget(r_is_taken);
                if (r_new_idx < 0) continue;
                
                int l_new_cs_idx = l_new_idx / {group_size};
                int r_new_cs_idx = r_new_idx / {group_size};
                
                {dtype} l_cur_acc = accs[l_new_cs_idx + l_old_idx * {chunk_size}];
                {dtype} r_cur_acc = accs[r_new_cs_idx + r_old_idx * {chunk_size}];
                
                {dtype} l_swap_acc = accs[r_new_cs_idx + l_old_idx * {chunk_size}];
                {dtype} r_swap_acc = accs[l_new_cs_idx + r_old_idx * {chunk_size}];
                
                if (l_swap_acc + r_swap_acc > l_cur_acc + r_cur_acc) {{
                    if (l_new_idx != atomicCAS(l_is_taken, l_new_idx, -1)) continue;
                    if (r_new_idx != atomicCAS(r_is_taken, r_new_idx, l_new_idx)) {{
                        vset(l_is_taken, l_new_idx);  // rollback
                        continue;
                    }}
                    vset(l_is_taken, r_new_idx);
                    something_swapped = 1;
                }}
                
            }}
        }}
        __global__ void kernel_postprocess2({dtype}* dense, {dtype}* sparse, int16_t* idx, int* is_taken_all) {{
            int64_t cs_idx = threadIdx.x + blockDim.x * blockIdx.x;
            int64_t os_idx = threadIdx.y + blockDim.y * blockIdx.y;
            int64_t gs_idx = threadIdx.z + blockDim.z * blockIdx.z;
            if (cs_idx >= {chunk_size}) return;
            if (os_idx >= {loop_outer_size}) return;
            if (gs_idx >= {group_size}) return;
            
            int* is_taken = is_taken_all + os_idx * {chunk_size} * {group_size};
            int16_t* idx_base = &idx[os_idx * {idx_loop_outer_stride}];
            {dtype}* dense_base = &dense[os_idx * {dense_loop_outer_stride}];
            {dtype}* sparse_base = &sparse[os_idx * {sparse_loop_outer_stride}];
            
            //printf("Applying final permutation...\\n");
            int old_idx = cs_idx * {group_size} + gs_idx;
            int new_idx = is_taken[old_idx];
            
            int new_cs_idx = new_idx / {group_size};
            
            // put into the output
            idx_base[new_idx * {idx_group_stride}] = old_idx;
            for (int64_t n_idx = 0; n_idx < {n}; n_idx++) {{
                int64_t m_idx = nnz_order[new_cs_idx][n_idx];
                sparse_base[n_idx + new_idx * {sparse_group_stride}] = dense_base[m_idx + old_idx * {dense_group_stride}];
            }}
        }}
        extern "C" void func({dtype}* dense, {dtype}* sparse, int16_t* idx, int* is_taken, {dtype}* accs, int* groups) {{
            something_swapped = 0;
            kernel_preprocess<<<dim3({grid_x}, {grid_y}, {grid_z}), dim3({block_x}, {block_y}, {block_z})>>>(dense, accs, is_taken, groups);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaStreamSynchronize(0));
            
            if (something_swapped) {{
                kernel_preprocess2<<<dim3({grid_x}, {grid_y}, {grid_z}), dim3({block_x}, {block_y}, {block_z})>>>(is_taken, groups);
                CUDA_CHECK(cudaPeekAtLastError());
            }}

            while (something_swapped) {{
                something_swapped = 0;
                kernel_postprocess<<<dim3({grid_x_pp}, {grid_y_pp}, {grid_z_pp}), dim3({block_x_pp}, {block_y_pp}, {block_z_pp})>>>(accs, is_taken);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cudaStreamSynchronize(0));
            }}
            
            kernel_postprocess2<<<dim3({grid_x}, {grid_y}, {grid_z}), dim3({block_x}, {block_y}, {block_z})>>>(dense, sparse, idx, is_taken);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaStreamSynchronize(0));
        }}
        """,
        lang="cu",
        # opts=["-G"],
        # opts=['--expt-relaxed-constexpr', '--expt-extended-lambda']
    )
    lib.func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func


def dense_to_grouped_n_m_merged(inp, n):
    impl_bulder = (
        get_dense_to_grouped_nm_impl_cpu
        if inp.device.type == "cpu"
        else get_dense_to_grouped_nm_impl_cuda
    )
    func = impl_bulder(
        inp.dtype,
        inp.shape,
        n,
    )
    loop_outer_size = inp.shape[0]
    chunk_size = inp.shape[1]
    group_size = inp.shape[2]
    sparse_val_shape = (*inp.shape[:-1], n)
    sparse_idx_shape = (*inp.shape[:-1], 1)
    out = torch.zeros(sparse_val_shape, dtype=inp.dtype, device=inp.device)
    idx = torch.full(
        sparse_idx_shape, fill_value=-1, dtype=torch.int16, device=inp.device
    )
    cont_inp = inp.contiguous()
    if inp.device.type == "cpu":
        func(cont_inp.data_ptr(), out.data_ptr(), idx.data_ptr())
    else:
        is_taken = torch.empty(
            loop_outer_size * chunk_size * group_size,
            dtype=torch.int,
            device=inp.device,
        )
        accs = torch.empty(
            loop_outer_size * chunk_size * chunk_size * group_size,
            dtype=inp.dtype,
            device=inp.device,
        )
        groups = torch.zeros(
            loop_outer_size * chunk_size, dtype=torch.int, device=inp.device
        )
        with torch.cuda.device(inp.device):
            # t1 = time.time()
            func(
                cont_inp.data_ptr(),
                out.data_ptr(),
                idx.data_ptr(),
                is_taken.data_ptr(),
                accs.data_ptr(),
                groups.data_ptr(),
            )
            # t2 = time.time()
            # print(f"dense->sparse kernel time {t2-t1:.2f}")
    return out, idx


def pad_to(tensor, new_shape):
    padding = [(0, p - s) for s, p in zip(tensor.shape, new_shape)]
    padding = [elem for pair in reversed(padding) for elem in pair]
    return torch.nn.functional.pad(tensor, padding)


def unpad_to(tensor, new_shape):
    assert tensor.ndim == len(new_shape)
    for d, l in enumerate(new_shape):
        tensor = tensor.narrow(dim=d, start=0, length=l)
    return tensor


def dense_to_grouped_n_m(tensor, n, m, sparse_dim, group_size, group_dim):
    nm_strides = compute_nm_strides(
        tensor.shape, n, m, sparse_dim, group_dim, group_size
    )

    padded = pad_to(tensor, nm_strides["padded_dense_shape"])
    expanded = padded.reshape(nm_strides["expanded_dense_shape"])
    permuted = expanded.permute(nm_strides["dim_permutation"])
    merged = permuted.reshape(nm_strides["merged_shape"])
    return (*dense_to_grouped_n_m_merged(merged, n), nm_strides)


@cache
def get_grouped_n_m_to_dense_impl_cpu(dense_dtype, val_shape, m):
    assert len(val_shape) == 4  # (batch_dim, chunk=m!/n!/(m-n)!, group, block=n)
    chunk_size = val_shape[1]
    group_size = val_shape[2]
    n = val_shape[3]
    assert chunk_size == fact(m) // fact(n) // fact(m - n)

    order = make_n_m(n, m)
    loop_outer_size = val_shape[0]

    dense_loop_outer_stride = chunk_size * group_size * m
    sparse_loop_outer_stride = chunk_size * group_size * n
    idx_loop_outer_stride = chunk_size * group_size * 1

    dense_group_stride = m
    sparse_group_stride = n
    idx_group_stride = 1

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    chunk_size = len(order)
    impl = compile(
        f"""
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        static const int8_t nnz_order[{chunk_size}][{n}] = {make_n_m_order_c(order)};
        extern "C" void func({dtype}* sparse, {dtype}* dense, int16_t* idx) {{
            for (int64_t os_idx = 0; os_idx < {loop_outer_size}; os_idx++) {{
                {dtype}* dense_base = &dense[os_idx * {dense_loop_outer_stride}];
                {dtype}* sparse_base = &sparse[os_idx * {sparse_loop_outer_stride}];
                int16_t* idx_base = &idx[os_idx * {idx_loop_outer_stride}];
                for (int64_t cs_idx = 0; cs_idx < {chunk_size}; cs_idx++) {{
                    for (int64_t gs_idx = 0; gs_idx < {group_size}; gs_idx++) {{
                        int16_t sparse_idx = (gs_idx + cs_idx * {group_size});
                        int16_t original_idx = idx_base[sparse_idx * {idx_group_stride}];
                        for (int64_t n_idx = 0; n_idx < {n}; n_idx++) {{
                            int64_t m_idx = nnz_order[cs_idx][n_idx];
                            dense_base[m_idx + original_idx * {dense_group_stride}] = sparse_base[n_idx + sparse_idx * {sparse_group_stride}];
                        }}
                    }}
                }}
            }}
        }}
        """,
    )
    impl.func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return impl.func


@cache
def get_grouped_n_m_to_dense_impl_cuda(dense_dtype, val_shape, m):
    assert len(val_shape) == 4  # (batch_dim, chunk=m!/n!/(m-n)!, group, block=n)
    chunk_size = val_shape[1]
    group_size = val_shape[2]
    n = val_shape[3]
    assert chunk_size == fact(m) // fact(n) // fact(m - n)

    order = make_n_m(n, m)
    loop_outer_size = val_shape[0]

    dense_loop_outer_stride = chunk_size * group_size * m
    sparse_loop_outer_stride = chunk_size * group_size * n
    idx_loop_outer_stride = chunk_size * group_size * 1

    dense_group_stride = m
    sparse_group_stride = n
    idx_group_stride = 1

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    chunk_size = len(order)

    block_x = 8
    block_y = 8
    block_z = 4

    grid_x = int_ceil(loop_outer_size, block_x)
    grid_y = int_ceil(chunk_size, block_y)
    grid_z = int_ceil(group_size, block_z)

    impl = compile(
        f"""
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <iostream>
        #define CUDA_CHECK(expr) do {{\\
            cudaError_t err = (expr);\\
            if (err != cudaSuccess) {{\\
                std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\\n"; \\
                abort(); \\
            }}\\
        }} while(0)
        __device__ static const int8_t nnz_order[{chunk_size}][{n}] = {make_n_m_order_c(order)};
        __global__ void kernel({dtype}* sparse, {dtype}* dense, int16_t* idx) {{
            int64_t os_idx = threadIdx.x + blockDim.x * blockIdx.x;
            int64_t cs_idx = threadIdx.y + blockDim.y * blockIdx.y;
            int64_t gs_idx = threadIdx.z + blockDim.z * blockIdx.z;
            if (os_idx >= {loop_outer_size}) return;
            if (cs_idx >= {chunk_size}) return;
            if (gs_idx >= {group_size}) return;
            {dtype}* dense_base = &dense[os_idx * {dense_loop_outer_stride}];
            {dtype}* sparse_base = &sparse[os_idx * {sparse_loop_outer_stride}];
            int16_t* idx_base = &idx[os_idx * {idx_loop_outer_stride}];
            int16_t sparse_idx = (gs_idx + cs_idx * {group_size});
            int16_t original_idx = idx_base[sparse_idx * {idx_group_stride}];
            for (int64_t n_idx = 0; n_idx < {n}; n_idx++) {{
                int64_t m_idx = nnz_order[cs_idx][n_idx];
                dense_base[m_idx + original_idx * {dense_group_stride}] = sparse_base[n_idx + sparse_idx * {sparse_group_stride}];
            }}
        }}
        extern "C" void func({dtype}* sparse, {dtype}* dense, int16_t* idx) {{
            kernel<<<dim3({grid_x}, {grid_y}, {grid_z}), dim3({block_x}, {block_y}, {block_z})>>>(sparse, dense, idx);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }}
        """,
        lang="cu",
    )
    impl.func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return impl.func


def grouped_n_m_to_dense(nm_strides, val, idx):
    impl_bulder = (
        get_grouped_n_m_to_dense_impl_cpu
        if val.device.type == "cpu"
        else get_grouped_n_m_to_dense_impl_cuda
    )
    func = impl_bulder(
        val.dtype,
        val.shape,
        nm_strides["m"],
    )

    out = torch.zeros(nm_strides["merged_shape"], dtype=val.dtype, device=val.device)
    if val.device.type == "cpu":
        func(val.data_ptr(), out.data_ptr(), idx.data_ptr())
    else:
        with torch.cuda.device(val.device):
            # t1 = time.time()
            func(val.data_ptr(), out.data_ptr(), idx.data_ptr())
            # t2 = time.time()
            # print(f"sparse->dense kernel time {t2-t1:.2f}")
    unmerged = out.reshape(nm_strides["permuted_shape"])
    unpermuted = unmerged.permute(nm_strides["inverse_dim_permutation"])
    unexpanded = unpermuted.reshape(nm_strides["padded_dense_shape"])
    unpadded = unpad_to(unexpanded, nm_strides["dense_shape"])

    return unpadded


class GroupedNMTensor:
    def __init__(self, val, idx, nm_strides):
        self.val = val
        self.idx = idx
        self.nm_strides = nm_strides

    @staticmethod
    def from_dense(tensor, n, m, sparse_dim, group_size, group_dim):
        val, idx, nm_strides = dense_to_grouped_n_m(
            tensor, n, m, sparse_dim, group_size, group_dim
        )
        ten = GroupedNMTensor(val, idx, nm_strides)
        ten.set_spmm_opt()
        return ten

    def set_spmm_opt(self, tile_a=None, acc_width=None, tile_b=None, kernel=None):
        flags = cpuinfo.get_cpu_info()["flags"]
        if "avx512f" in flags:
            kernel_autodetect = "avx512"
        else:
            kernel_autodetect = "avx2"

        self.nm_strides["tile_a"] = tile_a or 3
        self.nm_strides["acc_width"] = acc_width or 4
        self.nm_strides["tile_b"] = tile_b or 16
        self.nm_strides["kernel"] = kernel or kernel_autodetect

    def clone_layout(self, dense_tensor):
        raise NotImplementedError(
            "This function doesn't support proper device handling and should be removed"
        )
        # copies format and locations of nonzeros
        if dense_tensor.shape != self.nm_strides["dense_shape"]:
            raise ValueError(
                f"Shape mismatch: expected {self.nm_strides['dense_shape']}, received {dense_tensor.shape}"
            )

        exp_shape = self.nm_strides["expanded_dense_shape"]
        exp_group_dim = self.nm_strides["exp_group_dim"]
        exp_sparse_dim = self.nm_strides["exp_sparse_dim"]

        # step 1: reorder values but keep them non-sparsified
        padded_dense = pad_to(dense_tensor, self.nm_strides["padded_dense_shape"])
        exp_indexed_shape = (
            *exp_shape[: exp_group_dim + 1],
            exp_shape[exp_group_dim + 1] * exp_shape[exp_group_dim + 2],
            *exp_shape[exp_group_dim + 3 :],
        )
        broadcasted_idx = self.idx.to(device=dense_tensor.device).expand(exp_shape)
        padded_dense_for_reordering = padded_dense.reshape(exp_indexed_shape)
        broadcasted_idx_for_reordering = broadcasted_idx.reshape(exp_indexed_shape).to(
            torch.int64
        )
        reordered_dense = padded_dense_for_reordering.gather(
            dim=exp_group_dim + 1, index=broadcasted_idx_for_reordering
        )
        reordered_dense = reordered_dense.reshape(exp_shape)
        # step 2: drop values to match the nonzero mask
        order = torch.tensor(self.nm_strides["order"], device=dense_tensor.device)
        if exp_sparse_dim < exp_group_dim:
            order = order.t()
        singular_idx_shape = [1 for _ in exp_shape]
        singular_idx_shape[exp_group_dim + 1] = self.nm_strides["chunk_size"]
        singular_idx_shape[exp_sparse_dim + 1] = self.nm_strides["n"]
        sparse_indices = order.reshape(singular_idx_shape).expand_as(
            self.val.to(device=dense_tensor.device)
        )
        reordered_sparsified = reordered_dense.gather(
            dim=exp_sparse_dim + 1, index=sparse_indices
        )
        return GroupedNMTensor(
            reordered_sparsified.to(device="cpu"),
            self.idx.to(device="cpu"),
            self.nm_strides,
        )

    def to_dense(self):
        return grouped_n_m_to_dense(
            self.nm_strides,
            self.val,
            self.idx,
        )


class FixedMaskTensor:
    def __init__(self, val, mask, n, m, g):
        assert torch.all(
            torch.isclose(mask, torch.zeros_like(mask))
            | torch.isclose(mask, torch.ones_like(mask))
        )
        self.val = val
        self.mask = mask
        self.n = n
        self.m = m
        self.g = g

    @staticmethod
    def from_dense(tensor, n, m, g):
        mask = torch.where(
            tensor.abs() < 1e-6,
            torch.zeros_like(tensor, dtype=torch.bool),
            torch.ones_like(tensor, dtype=torch.bool),
        )
        return FixedMaskTensor(tensor * mask, mask, n, m, g)

    def to_dense(self):
        return copy.deepcopy(self.val)

    def numel(self):
        return self.val.numel()

    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        return FixedMaskTensor(
            self.val.to(device=device, dtype=dtype, copy=True),
            self.mask.to(device=device, dtype=dtype, copy=True),
            self.n,
            self.m,
            self.g,
        )

    @property
    def shape(self):
        return self.val.shape

    @property
    def device(self):
        return self.val.device

    @property
    def dtype(self):
        return self.val.dtype


class PerfectNMTensor:
    def __init__(self, val, idx, n, m, sparse_dim, sparse_dim_size):
        self.val = val
        self.idx = idx
        self.n = n
        self.m = m
        self.sparse_dim = sparse_dim
        self.sparse_dim_size = sparse_dim_size

    @staticmethod
    def from_dense(dense_tensor, n, m, sparse_dim):
        num_blocks = math.ceil(dense_tensor.shape[sparse_dim] / m)
        padded_sparse_dim = num_blocks * m
        padded_shape = [
            (padded_sparse_dim if idx == sparse_dim else dim)
            for idx, dim in enumerate(dense_tensor.shape)
        ]
        padded_tensor = pad_to(dense_tensor, padded_shape)
        expanded_shape = (
            dense_tensor.shape[:sparse_dim]
            + (num_blocks, m)
            + dense_tensor.shape[sparse_dim + 1 :]
        )
        expanded_tensor = padded_tensor.reshape(expanded_shape)
        sorted_indices = expanded_tensor.abs().argsort(
            dim=sparse_dim + 1, descending=True
        )
        sorted_vals = expanded_tensor.gather(dim=sparse_dim + 1, index=sorted_indices)
        sparse_vals = sorted_vals.narrow(dim=sparse_dim + 1, start=0, length=n)
        sparse_indices = sorted_indices.narrow(dim=sparse_dim + 1, start=0, length=n)
        return PerfectNMTensor(
            sparse_vals,
            sparse_indices,
            n,
            m,
            sparse_dim,
            dense_tensor.shape[sparse_dim],
        )

    def to_dense(self):
        exp_shape = self.val.shape
        exp_shape = (
            exp_shape[: self.sparse_dim + 1]
            + (self.m,)
            + exp_shape[self.sparse_dim + 2 :]
        )
        dense = torch.zeros(exp_shape, dtype=self.val.dtype, device=self.val.device)
        dense.scatter_(dim=self.sparse_dim + 1, index=self.idx, src=self.val)
        padded_shape = (
            exp_shape[: self.sparse_dim]
            + (exp_shape[self.sparse_dim] * self.m,)
            + exp_shape[self.sparse_dim + 2 :]
        )
        unpadded = dense.reshape(padded_shape).narrow(
            dim=self.sparse_dim, start=0, length=self.sparse_dim_size
        )
        return unpadded


def is_correct_nm(original_dense, sparsified_dense, sparse_dim, n, m):
    is_sparsified = (original_dense == sparsified_dense) | (
        sparsified_dense == torch.zeros_like(sparsified_dense)
    )
    if not is_sparsified.all():
        return False
    shape = original_dense.shape
    padded_shape = (
        shape[:sparse_dim]
        + (math.ceil(shape[sparse_dim] / m) * m,)
        + shape[sparse_dim + 1 :]
    )
    padded_sparsified = pad_to(sparsified_dense, padded_shape)
    expanded_shape = (
        shape[:sparse_dim]
        + (math.ceil(shape[sparse_dim] / m), m)
        + shape[sparse_dim + 1 :]
    )
    expanded_sparsified = padded_sparsified.reshape(expanded_shape)
    nnz_per_block = expanded_sparsified.bool().sum(dim=sparse_dim + 1)
    if not (nnz_per_block <= n).all():
        return False
    return True
