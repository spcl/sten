#!/usr/bin/env python

from textwrap import dedent
import subprocess
import jinja2
import itertools
import hashlib
import json
from math import factorial as fact
import math
import numpy as np
import sys
import pathlib
import argparse


def strides_from_shape(shape, custom={}):
    result = [custom[len(shape) - 1] if (len(shape) - 1) in custom else "1"]
    for i in reversed(range(0, len(shape) - 1)):
        current_stride = (
            custom[i] if (i in custom) else (shape[i + 1] + " * " + result[0])
        )
        result.insert(0, current_stride)
    return result


def size_from_shape(shape):
    return " * ".join(shape)


def linear_index(index, strides):
    assert len(index) == len(strides)
    return " + ".join(f"{i} * {s}" for i, s in zip(reversed(index), reversed(strides)))


def modify_access(access, replacements):
    for k, v in replacements.items():
        access = access.replace(k, str(v))
    return access


def is_increasing(l):
    return all([a < b for a, b in zip(l[:-1], l[1:])])


def is_adjacent(n1, n2):
    return sum([i1 != i2 for i1, i2 in zip(n1, n2)]) == 1


def find_path(sources, remaining, nodes):
    # nodes - list of available nodes
    # sources - indices of nodes which can be taken first
    # remaining - indices of nodes that can be used as targets
    targets_per_src = {}
    for src in sources:
        next_rem = remaining - {src}
        targets = set(tgt for tgt in next_rem if is_adjacent(nodes[src], nodes[tgt]))
        targets_per_src[src] = targets
    # heuristic: first try sources with the most targets
    targets_per_src = {
        k: v for k, v in sorted(targets_per_src.items(), key=lambda x: len(x[1]))
    }
    for src, targets in targets_per_src.items():
        next_rem = remaining - {src}
        for next_path in find_path(targets, next_rem, nodes):
            yield [nodes[src]] + next_path
    if not remaining:
        yield []


def make_blk_to_idx_list(m, n):
    assert m < n
    range_1 = list(range(n))
    range_n = [range_1[:] for _ in range(m)]
    nodes = [idx for idx in itertools.product(*range_n) if is_increasing(idx)]
    path = next(find_path(set(range(len(nodes))), set(range(len(nodes))), nodes))
    return path


def infer_strides(shape, stride):
    assert len(shape) == len(stride)
    # replaces None in stride assuming that current stride is equal to the current dimension
    stride = list(stride)
    for i in range(len(shape) - 1, -1, -1):
        if stride[i] is None:
            if i == len(shape) - 1:
                stride[i] = "1"
            else:
                stride[i] = f"{shape[i + 1]} * {stride[i + 1]}"
    return stride


def infer_reordered_shape(dims, pos_to_idx):
    n = len(dims)
    strides = [None for _ in range(n)]
    idx_to_pos = [None for i in range(n)]
    for pos, idx in enumerate(pos_to_idx):
        idx_to_pos[idx] = pos
    for idx in range(n - 1, -1, -1):
        pos = idx_to_pos[idx]
        if idx == n - 1:
            strides[pos] = "1"
        else:
            pos_next = idx_to_pos[idx + 1]
            strides[pos] = f"{dims[pos_next]} * {strides[pos_next]}"
    return list(zip(dims, strides))


class Array:
    def __init__(self, dtype, name, shape, align=None, base_offset=0):
        self.dtype = dtype
        self.name = name
        shape = [(x if isinstance(x, tuple) else (x, None)) for x in shape]
        self.dims, self.strides = zip(*shape)
        self.strides = infer_strides(self.dims, self.strides)
        self.align = align
        self.base_offset = base_offset

    def __getitem__(self, key):
        if isinstance(key, (int, str)):
            key = (key,)
        if isinstance(key, tuple):
            if all(isinstance(k, (str, int)) for k in key):
                if "" in key:
                    return self.subarray([(k if k != "" else None) for k in key])
                else:
                    return self.access(*key)
        raise IndexError(f"Can't get the item from array by key {key}")

    def __call__(self, *key):
        assert all(isinstance(k, tuple) for k in key)
        return self.tiled([(k if k != () else None) for k in key])

    def size(self):
        return math.prod(self.dims)

    def decl(self):
        if self.align is None:
            self.align = 1
        align_str = f" __attribute__ ((aligned ({self.align})))" if self.align else ""
        return f"{self.dtype} {self.name}[{self.size()}]{align_str};"
        return result

    def lin_idx(self, index):
        assert all([d != 0 for d in self.dims])
        if len(index) != len(self.strides):
            raise ValueError("Shape mismatch")
        res = " + ".join([f"{s} * {i}" for s, i in zip(self.strides, index)])
        res = f"{res} + {self.base_offset}"
        return res

    def access(self, *index):
        assert all([d != 0 for d in self.dims])
        return f"{self.name}[{self.lin_idx(index)}]"

    def subarray(self, index):
        assert all([d != 0 for d in self.dims])
        # index = ('dn2', 'dn3', 0, 'dk23_in_B', None, 0)
        assert len(index) == len(self.strides)
        partial_idx_list = [
            (i, s) for i, s in zip(index, self.strides) if i is not None
        ]
        partial_idx = " + ".join([f"{s} * {i}" for i, s in partial_idx_list])
        new_shape = [
            (d, s) for i, d, s in zip(index, self.dims, self.strides) if i is None
        ]
        return Array(
            self.dtype,
            self.name,
            new_shape,
            align=self.align,
            base_offset=f"{self.base_offset} + {partial_idx}",
        )

    def materialize(self, name):
        subarray_decl = f"{self.dtype}* {name} = &{self.name}[{self.base_offset}];"
        new_array = Array(
            self.dtype,
            name,
            zip(self.dims, self.strides),
            align=self.align,
            base_offset=0,
        )
        return subarray_decl, new_array

    def tiled(self, tiles):
        # original dim [M, N, K]
        # tiles [(M1, M2), (N1, N2, N3), None]
        assert len(tiles) == len(self.strides)
        new_shape = []
        for tile_shapes, last_stride, original_dim in zip(
            tiles, self.strides, self.dims
        ):
            if tile_shapes is None:
                tile_shapes = (original_dim,)
            tile_strides = [None for _ in tile_shapes]
            tile_strides[-1] = last_stride
            tile_strides = infer_strides(tile_shapes, tile_strides)
            new_shape += [(d, s) for d, s in zip(tile_shapes, tile_strides)]
        return Array(
            self.dtype,
            self.name,
            new_shape,
            align=self.align,
            base_offset=self.base_offset,
        )

    def offset(self, *offsets):
        assert len(offsets) == len(self.dims)
        new_base_offset = self.lin_idx(offsets)
        new_shape = [
            (f"({d} - {o})", s) for d, o, s in zip(self.dims, offsets, self.strides)
        ]
        return Array(
            self.dtype,
            self.name,
            new_shape,
            align=self.align,
            base_offset=new_base_offset,
        )

    def indexed_ofset(self, dim_idx, offset):
        offsets = [0 for _ in self.dims]
        offsets[dim_idx] = offset
        return self.offset(*offsets)

    def gtile(self, dim_idx, tile_size):
        assert all([d != 0 for d in self.dims])
        N = self.dims[dim_idx]
        N2 = tile_size
        N1 = N // N2
        N2H = N % N2

        A_shape = list(zip(self.dims, self.strides))
        A_stride = self.strides[dim_idx]

        AF_shape = (
            A_shape[:dim_idx]
            + [(N1, f"{A_stride} * {N2}"), (N2, A_stride)]
            + A_shape[dim_idx + 1 :]
        )
        AR_shape = (
            A_shape[:dim_idx]
            + [(1, f"{A_stride} * {N2}"), (N2H, A_stride)]
            + A_shape[dim_idx + 1 :]
        )

        AF = Array(
            self.dtype,
            self.name,
            AF_shape,
            align=self.align,
            base_offset=self.base_offset,
        )

        AR = self.indexed_ofset(dim_idx, N1 * N2)
        AR = Array(
            self.dtype,
            self.name,
            AR_shape,
            align=self.align,
            base_offset=AR.base_offset,
        )

        # assert all([d != 0 for d in AF.dims])
        # assert all([d != 0 for d in AR.dims])

        # return full array, remainder array, full size, remainder size
        return AF, AR, N1, N2H

    def shorten(self, *new_dims):
        # new_dims: [None, 10, None, 20, None]
        assert len(new_dims) == len(self.dims)

        new_dims = list(new_dims)
        for i, (old_dim, new_dim) in enumerate(zip(self.dims, new_dims)):
            if new_dim is not None:
                assert new_dim <= old_dim
            else:
                new_dims[i] = old_dim
        return Array(
            self.dtype,
            self.name,
            list(zip(new_dims, self.strides)),
            align=self.align,
            base_offset=self.base_offset,
        )


class IdGenerator:
    def __init__(self):
        self.counters = {}

    def make(self, name):
        id = self.counters.get(name, 0)
        self.counters[name] = id + 1
        return f"{name}_{id}"


def loop_nest(dims, reorder=None):
    # dims = [('dn2', 'DN2'), ('dm2', 'DM2') ('dn3', 'DN3')]
    # reorder = [0, 2, 1]
    if reorder is None:
        reorder = list(range(len(dims)))
    reordered_dims = [dims[i] for i in reorder]
    return " ".join([f"FOR({idx}, {num})" for idx, num in reordered_dims])


def stringify(params):
    if isinstance(params, dict):
        return "_".join([stringify(k) + "_" + stringify(v) for k, v in params.items()])
    elif isinstance(params, str):
        return params
    elif isinstance(params, list):
        return "_".join([stringify(p) for p in params])
    else:
        res = str(params)
        if " " in res:
            raise ValueError(f"Stringified <{res}> contains not allowed symbols")
        return res


def make_template(x):
    return jinja2.Template(
        dedent(x),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )


def transpose_4x4(A, B):
    return make_template(
        """\
        __m128 x0 = _mm_loadu_ps(&{{ A.access(0, 0) }});
        __m128 x1 = _mm_loadu_ps(&{{ A.access(1, 0) }});
        __m128 x2 = _mm_loadu_ps(&{{ A.access(2, 0) }});
        __m128 x3 = _mm_loadu_ps(&{{ A.access(3, 0) }});
        __m128 y0 = _mm_unpacklo_ps(x0, x1);
        __m128 y1 = _mm_unpackhi_ps(x0, x1);
        __m128 y2 = _mm_unpacklo_ps(x2, x3);
        __m128 y3 = _mm_unpackhi_ps(x2, x3);
        __m128 z0 = _mm_movelh_ps(y0, y2);
        __m128 z1 = _mm_movehl_ps(y2, y0);
        __m128 z2 = _mm_movelh_ps(y1, y3);
        __m128 z3 = _mm_movehl_ps(y3, y1);
        _mm_store_ps(&{{ B.access(0, 0) }}, z0);
        _mm_store_ps(&{{ B.access(1, 0) }}, z1);
        _mm_store_ps(&{{ B.access(2, 0) }}, z2);
        _mm_store_ps(&{{ B.access(3, 0) }}, z3);
    """
    ).render({**globals(), **locals()})


def dense_gemm_kernel(A1, B1, C, params):
    # requires DK2, DK3, DN4, DN5, DM3
    # A1 [DK2, DK3, DM3]
    # B1 [DK2, DK3, DN4, DN5]
    # C [DM3, DN4, DN5]
    DK2 = params[A1.dims[0]]
    DK3 = params[A1.dims[1]]
    DN4 = params[B1.dims[2]]
    DN5 = params[B1.dims[3]]
    DM3 = params[A1.dims[2]]
    assert DM3 % 2 == 0
    return make_template(
        """\
        {% for dm3 in range(DM3) %}
        {% for dn4 in range(DN4) %}
        __m256 acc_{{ dm3 }}_{{ dn4 }} = _mm256_setzero_ps();
        {% endfor %}
        {% endfor %}
        FOR (dk2, DK2) {
            {% for dk3 in range(DK3) %}
            {
                {% for dn4 in range(DN4) %}
                __m256 vb_{{ dn4 }} = _mm256_loadu_ps(&{{ B1.access('dk2', dk3, dn4, 0) }});
                __m256 vb_0_{{ dn4 }} = _mm256_moveldup_ps(vb_{{ dn4 }});
                __m256 vb_1_{{ dn4 }} = _mm256_movehdup_ps(vb_{{ dn4 }});
                {% endfor %}
                {% for dm3 in range(DM3 // 2) %}
                {
                    __m256 va = (__m256)_mm256_broadcast_sd((double*)&{{ A1.access('dk2', dk3, dm3 * 2) }});
                    {% for dn4 in range(DN4) %}
                    acc_{{ dm3 * 2 + 0 }}_{{ dn4 }} = _mm256_fmadd_ps(va, vb_0_{{ dn4 }}, acc_{{ dm3 * 2 + 0 }}_{{ dn4 }});
                    acc_{{ dm3 * 2 + 1 }}_{{ dn4 }} = _mm256_fmadd_ps(va, vb_1_{{ dn4 }}, acc_{{ dm3 * 2 + 1 }}_{{ dn4 }});
                    {% endfor %}
                }
                {% endfor %}
            }
            {% endfor %}
        }
        {% for dm3 in range(DM3 // 2) %}
        {% for dn4 in range(DN4) %}
        {
            float* addr0 = &{{ C.access(dm3 * 2 + 0, dn4, 0) }};
            float* addr1 = &{{ C.access(dm3 * 2 + 1, dn4, 0) }};
            
            __m256 x0 = acc_{{ dm3 * 2 + 0 }}_{{ dn4 }};
            __m256 x1 = acc_{{ dm3 * 2 + 1 }}_{{ dn4 }};
            __m256 y0 = _mm256_unpacklo_ps(x0, x1);
            __m256 y1 = _mm256_unpackhi_ps(x0, x1);
            __m256 z0 = (__m256)_mm256_unpacklo_pd((__m256d) y0, (__m256d) y1);
            __m256 z1 = (__m256)_mm256_unpackhi_pd((__m256d) y0, (__m256d) y1);

            _mm256_storeu_ps(addr0, _mm256_add_ps(_mm256_loadu_ps(addr0), z0));
            _mm256_storeu_ps(addr1, _mm256_add_ps(_mm256_loadu_ps(addr1), z1));
        }
        {% endfor %}
        {% endfor %}
    """
    ).render({**globals(), **locals()})


def make_load_mask(m, n):
    assert m < n
    return ", ".join([("-1" if i < m else "0") for i in range(n)])


def params_hash(params):
    name = stringify(params)
    return hashlib.sha256(name.encode()).hexdigest()[:7]


def benchmark_parametrized_dense_dense(params):
    name = stringify(params)

    encoded_name = params_hash(params)

    print(f"Configuration hash: {encoded_name} Params: {json.dumps(params)}")

    signature = f"void dense_dense_{encoded_name}(int64_t M, int64_t N, int64_t K, float* __restrict__ A, int64_t lda, float* __restrict__ B, int64_t ldb, float* __restrict__ C, int64_t ldc)"

    target_M = params["target_M"]
    target_N = params["target_N"]

    DK = params["DK"]
    DK23 = params["DK2"] * params["DK3"]
    DK3 = params["DK3"]

    DM3 = params["DM3"]

    DM4C = 4
    assert DM3 % DM4C == 0
    DM3C = DM3 // DM4C

    DK4C = 4
    DK3C = 1
    # assert (DK23) % (DK3C * DK4C) == 0
    DK2C = (DK23) // (DK3C * DK4C)

    DM2 = params["DM2"]

    DN5 = params["DN5"]
    DN4 = params["DN4"]
    DN3 = params["DN3"]
    DN2 = params["DN2"]

    DM1_ = DM2 * DM3
    DN1_ = DN2 * DN3 * DN4 * DN5

    DM1 = max(round(target_M * 1.0 / DM1_), 1)
    DN1 = max(round(target_N * 1.0 / DN1_), 1)

    M = DM1 * DM2 * DM3
    N = DN1 * DN2 * DN3 * DN4 * DN5
    K = params["DK"]

    loop_order_copy_b = params["loop_order_copy_b"]
    loop_order_copy_a = params["loop_order_copy_a"]
    loop_order_inner = params["loop_order_inner"]

    B1_layout = params["B1_layout"]
    A1_layout = params["A1_layout"]

    id_gen = IdGenerator()

    template = jinja2.Template(
        dedent(
            """\
        // WARNING! THIS IS GENERATED FILE! DO NOT EDIT!

        #include "dense_dense_{{ encoded_name }}.h"

        #include <immintrin.h>
        #include <stdio.h>
        #include <string.h>
        #include <malloc.h>
        
        #define FOR(i, n) for (int64_t i = 0; i < (n); i++)
        
        {% macro transpose_4x4_(A, B) %}
            __m128 x0 = _mm_loadu_ps(&{{ A[0, 0] }});
            __m128 x1 = _mm_loadu_ps(&{{ A[1, 0] }});
            __m128 x2 = _mm_loadu_ps(&{{ A[2, 0] }});
            __m128 x3 = _mm_loadu_ps(&{{ A[3, 0] }});
            __m128 y0 = _mm_unpacklo_ps(x0, x1);
            __m128 y1 = _mm_unpackhi_ps(x0, x1);
            __m128 y2 = _mm_unpacklo_ps(x2, x3);
            __m128 y3 = _mm_unpackhi_ps(x2, x3);
            __m128 z0 = _mm_movelh_ps(y0, y2);
            __m128 z1 = _mm_movehl_ps(y2, y0);
            __m128 z2 = _mm_movelh_ps(y1, y3);
            __m128 z3 = _mm_movehl_ps(y3, y1);
            _mm_store_ps(&{{ B[0, 0] }}, z0);
            _mm_store_ps(&{{ B[1, 0] }}, z1);
            _mm_store_ps(&{{ B[2, 0] }}, z2);
            _mm_store_ps(&{{ B[3, 0] }}, z3);
        {% endmacro %}

        {% macro kernel_acc_mn_vec_(A, B) %}{
            {% set DM, DN, VEC = A.dims[0], B.dims[0], 8 %}
            {% set DN1, DN2R = math.ceil(DN / VEC), DN % VEC %}
            {% for dn1 in range(DN1) %}
                {% if (dn1 < DN1 - 1) or (DN2R == 0) %}
                    __m256 vb_{{ dn1 }} = _mm256_loadu_ps(&{{ B[dn1 * VEC] }});
                {% else %}
                    __m256 vb_{{ dn1 }} = _mm256_maskload_ps(&{{ B[dn1 * VEC] }}, load_mask);
                {% endif %}
            {% endfor %}
            {% for dm in range(DM) %}{
                __m256 va = _mm256_broadcast_ss(&{{ A[dm] }});
                {% for dn1 in range(DN1) %}{
                    acc_{{ dm }}_{{ dn1 }} = _mm256_fmadd_ps(va, vb_{{ dn1 }}, acc_{{ dm }}_{{ dn1 }});
                }{% endfor %}
            }{% endfor %}
        }{% endmacro %}

        {% macro dense_gemm_kernel_(A1, B1, C) %}{
            {% set DK23, DN45, DM3, VEC = A1.dims[0], B1.dims[1], A1.dims[1], 8 %}
            {% set DN4, DN5R = math.ceil(DN45 / VEC), (DN45 % VEC) %}
            // init accumulators
            {% for dm3 in range(DM3) %}
                {% for dn4 in range(DN4) %}
                    __m256 acc_{{ dm3 }}_{{ dn4 }} = _mm256_setzero_ps();
                {% endfor %}
            {% endfor %}
            {% if DN5R != 0 %}__m256i load_mask = _mm256_setr_epi32({{ make_load_mask(DN5R, VEC) }});{% endif %}
            {% set A1F, A1R, DK2F, DK3R = A1.gtile(0, params["DK3"]) %}
            {% set B1F, B1R, DK2F, DK3R = B1.gtile(0, params["DK3"]) %}
            {% set dk2 = id_gen.make("dk2") %}
            FOR ({{ dk2 }}, {{ DK2F }}) {
                {% for dk3 in range(DK3) %}{
                    {{ kernel_acc_mn_vec_(A1F[dk2, dk3, ''], B1F[dk2, dk3, '']) }}
                }{% endfor %}
            }
            {% if DK3R != 0 %}{
                {% for dk3 in range(DK3R) %}{
                    {{ kernel_acc_mn_vec_(A1R[0, dk3, ''], B1R[0, dk3, '']) }}
                }{% endfor %}
            }{% endif %}
            {% for dm3 in range(DM3) %}
                {% for dn4 in range(DN4) %}{
                    float* addr = &{{ C[dm3, dn4 * VEC] }};
                    {% if (dn4 < DN4 - 1) or (DN5R == 0) %}
                        _mm256_storeu_ps(addr, _mm256_add_ps(_mm256_loadu_ps(addr), acc_{{ dm3 }}_{{ dn4 }}));
                    {% else %}
                        _mm256_maskstore_ps(addr, load_mask, _mm256_add_ps(_mm256_maskload_ps(addr, load_mask), acc_{{ dm3 }}_{{ dn4 }}));
                    {% endif %}
                }{% endfor %}
            {% endfor %}
        }{% endmacro %}
        
        {% macro dense_gemm_kernel_simple_(A1, B1, C) %}{
            {% set DK2, DK3, DN45, DM3 = A1.dims[0], A1.dims[1], B1.dims[2], A1.dims[2] %}            
            FOR(dk2, {{ DK2 }}) FOR(dk3, {{ DK3 }}) FOR(dm3, {{ DM3 }}) FOR(dn45, {{ DN45 }}) {
                {{ C['dm3', 0, 'dn45'] }} += {{ A1['dk2', 'dk3', 'dm3'] }} * {{ B1['dk2', 'dk3', 'dn45'] }};
            }
        }{% endmacro %}

        {{ signature }} {

            const int64_t DN5 = {{ DN5 }};
            const int64_t DN4 = {{ DN4 }};
            const int64_t DN3 = {{ DN3 }};
            const int64_t DN2 = {{ DN2 }};

            const int64_t DM4C = {{ DM4C }};
            const int64_t DM3C = {{ DM3C }};
                        
            const int64_t DM3 = {{ DM3 }};

            const int64_t DM2 = {{ DM2 }};

            const int64_t DK4C = {{ DK4C }};
            const int64_t DK3C = {{ DK3C }};
            const int64_t DK2C = {{ DK2C }};

            {% set A = Array('float', 'A', [(M, 'lda'), K]) %}

            {% set B = Array('float', 'B', [(K, 'ldb'), N]) %}
            {% set C = Array('float', 'C', [DM1 * DM2, (DM3, 'ldc'), DN1, DN2 * DN3, DN4 * DN5]) %}
            
            {% set B1 = Array('float', 'B1', [DN2 * DN3, DK23, DN4 * DN5], align=32) %}
            {% set A1 = Array('float', 'A1', [DM1 * DM2, DK23, DM3], align=32) %}
            
            {{ A1.decl() }}
            {{ B1.decl() }}
            
            {% macro process_dk1_(A, B, C, A1, B1) %}
                {% set DK23 = A.dims[1] %}
                {% set dn1 = id_gen.make("dn1") %}
                FOR ({{ dn1 }}, {{ DN1 }}) {
                    // copy tile of B into higher level of cache
                    {% macro copy_B(B, B1) %}
                        {% set N23, K23, N45 = B1.dims %}
                        {% set dk23, dn23, dn45 = id_gen.make("dk23"), id_gen.make("dn23"), id_gen.make("dn45") %}
                        #pragma omp for
                        FOR({{ dk23 }}, {{ K23 }}) FOR({{ dn23 }}, {{ N23 }}) FOR({{ dn45 }}, {{ N45 }}) {
                            {{ B1[dn23, dk23, dn45] }} = {{ B((), (N23, N45))[dk23, dn23, dn45] }};
                        }
                    {% endmacro %}
                    #pragma omp parallel
                    {
                    {{ copy_B(B((), (DN1, DN1_))['', dn1, ''], B1) }}
                    if ({{ dn1 }} == 0) {
                        {% set dm1 = id_gen.make("dm1") %}
                        #pragma omp for
                        FOR ({{ dm1 }}, {{ DM1 }}) {
                            // copy tile of A into higher level of cache
                            {% set dm2, dm3, dk23 = id_gen.make("dm2"), id_gen.make("dm3"), id_gen.make("dk23") %}
                            FOR({{ dm2 }}, {{ DM2 }}) FOR({{ dm3 }}, {{ DM3 }}) FOR({{ dk23 }}, {{ DK23 }}) { 
                                {{ A1((DM1, DM2), (), ())[dm1, dm2, dk23, dm3] }} = {{ A((DM1, DM2, DM3), ())[dm1, dm2, dm3, dk23] }};
                            }
                            // compute
                            {% set dm2, dn23 = id_gen.make("dm2"), id_gen.make("dn23") %}
                            FOR({{ dm2 }}, {{ DM2 }}) FOR({{ dn23 }}, {{ DN2 * DN3 }}) {
                                {{ dense_gemm_kernel_(A1((DM1, DM2), (), ())[dm1, dm2, '', ''],
                                                    B1[dn23, '', ''],
                                                    C((DM1, DM2), (), (), (), ())[dm1, dm2, '', dn1, dn23, '']) }}
                            }
                        }
                    } else {
                        // compute
                        {% set dm12, dn23 = id_gen.make("dm12"), id_gen.make("dn23") %}
                        #pragma omp for
                        FOR({{ dm12 }}, {{ DM1 * DM2 }}) FOR({{ dn23 }}, {{ DN2 * DN3 }}) {
                            {{ dense_gemm_kernel_(A1[dm12, '', ''],
                                                B1[dn23, '', ''],
                                                C[dm12, '', dn1, dn23, '']) }}
                        }
                    }
                    } // pragma omp parallel
                }
            {% endmacro %}
            {% set AF, AR, DK1, DK2R = A.gtile(1, DK23) %}
            {% set BF, BR, DK1, DK2R = B.gtile(0, DK23) %}
            {% set dk1 = id_gen.make("dk1") %}
            FOR({{ dk1 }}, {{ DK1 }}) {
                {{ process_dk1_(AF['', dk1, ''], BF[dk1, '', ''], C, A1, B1) }}
            }
            {% if DK2R != 0 %}{
                {{ process_dk1_(AR['', 0, ''], BR[0, '', ''], C, A1.shorten(None, DK2R, None), B1.shorten(None, DK2R, None)) }}
            }{% endif %}
        }
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    code = template.render({**globals(), **locals()})

    template = jinja2.Template(
        dedent(
            """\
        // WARNING! THIS IS GENERATED FILE! DO NOT EDIT!
        
        #pragma once
        
        #include <stdint.h>
        
        #ifdef __cplusplus
        extern "C" {
        #endif
        
        /*
        Configuration:
        {{ json.dumps(params) }}
        */

        {{ signature }};

        #ifdef __cplusplus
        } // extern "C"
        #endif
    """
        )
    )
    header = template.render({**globals(), **locals()})

    with open(f"generated/dense_dense_{encoded_name}.c", "w") as f:
        f.write(code)
    with open(f"generated/dense_dense_{encoded_name}.h", "w") as f:
        f.write(header)

    main_template = jinja2.Template(
        dedent(
            """\
        #include "openblas/cblas.h"

        #include "common.hpp"

        #include "generated/dense_dense_{{ encoded_name }}.h"

        int main() {
            int64_t M = {{ DM1 * DM2 * DM3 }};
            int64_t K = {{ DK }};
            int64_t N = {{ DN1 * DN2 * DN3 * DN4 * DN5 }};

            std::vector<float> dA = rand_gen(M * K);
            std::vector<float> dB = rand_gen(K * N);
            
            std::vector<float> dCd(M * N);

            std::vector<float> dCd_ref(M * N);
            
            moment t1, t2;
            
            {
                t1 = timer::now();
                cblas_sgemm(
                    CblasRowMajor, // CBLAS_LAYOUT layout,
                    CblasNoTrans, // CBLAS_TRANSPOSE TransA,
                    CblasNoTrans, // CBLAS_TRANSPOSE TransB,
                    M, // const CBLAS_INDEX M,
                    N, // const CBLAS_INDEX N,
                    K, // const CBLAS_INDEX K,
                    1.0, // const float alpha,
                    dA.data(), // const float *A,
                    K, // const CBLAS_INDEX lda,
                    dB.data(), // const float *B,
                    N, // const CBLAS_INDEX ldb,
                    0.0, // const float beta,
                    dCd_ref.data(), // float *C,
                    N // const CBLAS_INDEX ldc
                );
                t2 = timer::now();
                printf("openblas_dense_dense seconds %.3g ns_per_fma %.3g\\n", seconds(t2 - t1), seconds(t2 - t1) / (M * N * K) * 1e9);
            }

            {
                std::fill(dCd.begin(), dCd.end(), 0.0f);
                t1 = timer::now();
                dense_dense_{{ encoded_name }}(
                    M, // const CBLAS_INDEX M,
                    N, // const CBLAS_INDEX N,
                    K, // const CBLAS_INDEX K,
                    dA.data(), // const float *A,
                    K, // const CBLAS_INDEX lda,
                    dB.data(), // const float *B,
                    N, // const CBLAS_INDEX ldb,
                    dCd.data(), // float *C,
                    N // const CBLAS_INDEX ldc
                );
                t2 = timer::now();
                printf("my_dense_dense seconds %.3g ns_per_fma %.3g\\n", seconds(t2 - t1), seconds(t2 - t1) / (M * N * K) * 1e9);
                CHECK(vector_allclose(dCd, dCd_ref));
            }
        }
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    main = main_template.render({**globals(), **locals()})

    with open(f"generated/dense_dense_{encoded_name}_main.cpp", "w") as f:
        f.write(main)

    subprocess.run(
        f"gcc -march=native -g -O3 -fopenmp -lpthread generated/dense_dense_{encoded_name}.c -c -o generated/dense_dense_{encoded_name}.o".split(),
        check=True,
    )
    subprocess.run(
        f"g++ -march=native -g -O3 -fopenmp -lpthread -I. -I./OpenBLAS/install/include generated/dense_dense_{encoded_name}_main.cpp ./OpenBLAS/install/lib/libopenblas.a -lpthread generated/dense_dense_{encoded_name}.o -o generated/dense_dense_{encoded_name}_main".split(),
        check=True,
    )
    subprocess.run(f"generated/dense_dense_{encoded_name}_main", check=True)


def generate_parametrized_sparse_dense(params):
    params = derive_params(params)
    encoded_name = params_hash(params)

    print(
        f"Configuration hash: {encoded_name} Params: {json.dumps(params)}",
        file=sys.stderr,
    )

    signature = f"void sparse_dense_{encoded_name}(float* __restrict__ A_val, int16_t* __restrict__ A_idx, float* __restrict__ B, float* __restrict__ C)"

    DK3 = params["DK3"]  #
    DK2 = (
        fact(params["DM3"])
        // fact(params["DM3S"])
        // fact(params["DM3"] - params["DM3S"])
    )  # number of sparse groups
    params["DK2"] = DK2

    DM3 = params["DM3"]  # accumulator tile size (should be 4 for 4:2 pattern)
    DM3S = params["DM3S"]  # sparse tile size
    DM2 = params["DM2"]  # cache tile size

    DN5 = params["DN5"]  # vector size
    DN4 = params["DN4"]  # accumulator tile size
    DN3 = params["DN3"]  # unused for now
    DN2 = params["DN2"]  # cache tile size
    params["DN45"] = DN4 * DN5
    params["DM23S"] = DM2 * DM3S
    params["DM23"] = DM2 * DM3
    params["DN2345"] = DN2 * DN3 * DN4 * DN5
    params["DK23"] = DK2 * DK3

    DN = params["DN"]
    DK = params["DK"]
    DM = params["DM"]
    bti = make_blk_to_idx_list(DM3S, DM3)

    params["DMI_padded"] = math.ceil(DM / DM3)
    params["DM_padded"] = params["DMI_padded"] * DM3
    params["DMS_padded"] = params["DMI_padded"] * DM3S

    params["DK_padded"] = math.ceil(DK / (DK2 * DK3)) * (DK2 * DK3)

    bti_str = (
        "{"
        + ", ".join(
            "{" + ", ".join(str(idx) for idx in idx_tuple) + "}" for idx_tuple in bti
        )
        + "}"
    )

    # strides
    SAMS = params["SAMS"]
    SAKS = params["SAKS"]
    SAMI = params["SAMI"]
    SAKI = params["SAKI"]
    SBK = params["SBK"]
    SBN = params["SBN"]
    SCM = params["SCM"]
    SCN = params["SCN"]

    id_gen = IdGenerator()

    template = jinja2.Template(
        dedent(
            """\
        // WARNING! THIS IS GENERATED FILE! DO NOT EDIT!

        #include "sparse_dense_{{ encoded_name }}.h"

        #include <immintrin.h>
        #include <stdio.h>
        #include <string.h>
        #include <malloc.h>
        
        #define FOR(i, n) for (int64_t i = 0; i < (n); i++)
        
        {{ signature }} {

            {% set A_val = Array('float', 'A_val', [(params["DMS_padded"], SAMS), (params["DK_padded"], SAKS)]) %}
            {% set A_idx = Array('int16_t', 'A_idx', [(params["DMI_padded"], SAMI), (params["DK_padded"], SAKI)]) %}
            // A_val.dims {{ A_val.dims }} A_idx.dims {{ A_idx.dims }}

            {% set B = Array('float', 'B', [(DK, SBK), (DN, SBN)]) %}
            {% set C = Array('float', 'C', [(DM, SCM), (DN, SCN)]) %}
            // C.dims {{ C.dims }} C.strides {{ C.strides }}

            {% set B1 = Array('float', 'B1', [DN2 * DN3, DK2 * DK3, DN4 * DN5], align=32) %}
            {% set A1_val = Array('float', 'A1_val', [params["DMS_padded"], DK2 * DK3], align=32) %}
            {% set A1_idx = Array('int16_t', 'A1_idx', [params["DMI_padded"], DK2 * DK3], align=32) %}

            {% macro sparse_dense_gemm_kernel_(A1_val, A1_idx, B1, C, in_remainder_DK2) %}
                {% set DM3S_padded, DK23_padded = A1_val.dims %}
                {% set (DK23_padded,) = A1_idx.dims %}
                {% set DK23, DN45 = B1.dims %}
                {% set DM3, DN45 = C.dims %}
                {% set DK2 = DK23_padded // params['DK3'] %}
                {% set DN4, DN5R = math.ceil(DN45 / params['DN5']), DN45 % params['DN5'] %}
                // A1_val.dims {{ A1_val.dims }} A1_idx.dims {{ A1_idx.dims }} B1.dims {{ B1.dims }}
                // init accumulators
                {% for dm3s in range(DM3S_padded) %}
                {% for dn4 in range(DN4) %}
                __m256 acc_{{ dm3s }}_{{ dn4 }} = _mm256_setzero_ps();
                {% endfor %}
                {% endfor %}
                {% if DN5R != 0 %}__m256i load_mask = _mm256_setr_epi32({{ make_load_mask(DN5R, DN5) }});{% endif %}
                {% for dk2 in range(DK2) %}{
                    {% set si = bti[dk2] %}
                    {% set dk3 = id_gen.make("dk3") %}
                    {% set in_remainder_DK2 = (DK23_padded != DK23) %}
                    // in_remainder_DK2 = {{ in_remainder_DK2 }} dk2 {{ dk2 }} DK2 {{ DK2 }} DK23_padded {{ DK23_padded }} DK23 {{ DK23 }}
                    FOR({{dk3}}, {{DK3}}) {
                        {% set dk23 = id_gen.make("dk23") %}
                        int {{dk23}} = {{ dk2 }} * {{ DK3 }} + {{ dk3 }};
                        {% set dk23_in_B = id_gen.make("dk23_in_B") %}
                        int16_t {{dk23_in_B}} = {{ A1_idx[dk23] }};
                        {% if in_remainder_DK2 %}
                            // in dk2 remainder
                            if ({{dk23_in_B}} < 0) break;
                        {% endif %}
                        {%+ set DECL, B2 = B1[dk23_in_B, ''].materialize('B2') %}{{ DECL }}
                        {% for dm3s in range(DM3S_padded) %}
                            __m256 va_{{ dm3s }} = _mm256_broadcast_ss(&{{ A1_val[dm3s, dk23] }});
                        {% endfor %}
                        {% for dn4 in range(DN4) %}{
                            {% set in_remainder_DN5 = (dn4 == DN4 - 1) and (DN5R != 0) %}
                            {% if not in_remainder_DN5 %}
                                __m256 vb_{{ dn4 }} = _mm256_loadu_ps(&{{ B2[dn4 * DN5] }});
                            {% else %}
                                __m256 vb_{{ dn4 }} = _mm256_maskload_ps(&{{ B2[dn4 * DN5] }}, load_mask);
                            {% endif %}
                            {% for dm3s in range(DM3S_padded) %}
                                acc_{{ dm3s }}_{{ dn4 }} = _mm256_fmadd_ps(va_{{ dm3s }}, vb_{{ dn4 }}, acc_{{ dm3s }}_{{ dn4 }});
                            {% endfor %}
                        }{% endfor %}
                    }
                    {% for dm3s in range(DM3S_padded) %}
                        {% if (dk2 == DK2 - 1) or (bti[dk2][dm3s] != bti[dk2+1][dm3s]) %}
                            {% for dn4 in range(DN4) %}{
                                {% set in_remainder_DN5 = (dn4 == DN4 - 1) and (DN5R != 0) %}
                                {% set dm3 = si[dm3s] %}
                                {% if dm3 < DM3 %} // cut out remainder DM3
                                    {% if SCN == 1 %}
                                        float* c_addr = &{{ C[dm3, dn4 * DN5] }};
                                        {% if not in_remainder_DN5 %}
                                            _mm256_storeu_ps(c_addr, _mm256_add_ps(_mm256_loadu_ps(c_addr), acc_{{ dm3s }}_{{ dn4 }}));
                                        {% else %}
                                            _mm256_maskstore_ps(c_addr, load_mask, _mm256_add_ps(_mm256_maskload_ps(c_addr, load_mask), acc_{{ dm3s }}_{{ dn4 }}));
                                        {% endif %}
                                    {% else %}
                                        {% set dn5 = id_gen.make("dn5") %}
                                        FOR ({{dn5}}, {{DN5R if in_remainder_DN5 else DN5}}) {
                                            float* acc_ptr = (float*)&acc_{{ dm3s }}_{{ dn4 }};
                                            {{ C((), (DN4, DN5))[dm3, dn4, dn5] }} += acc_ptr[{{dn5}}];
                                        }
                                    {% endif %}
                                {% endif %}
                                {% if (dk2 != DK2 - 1) %}
                                    acc_{{ dm3s }}_{{ dn4 }} = _mm256_setzero_ps();
                                {% endif %}
                            }{% endfor %}
                        {% endif %}
                    {% endfor %}
                }{% endfor %}
            {% endmacro %}
            
            {% macro copy_A(A1, A) %}
                {% set DM2, DK23 = A.dims %}
                // copy tile of A into higher level of cache
                {% set dm2 = id_gen.make("dm2") %}
                {% set dk23 = id_gen.make("dk23") %}
                FOR({{dm2}}, {{DM2}}) FOR({{dk23}}, {{DK23}}) {
                    {{ A1[dm2, dk23] }} = {{ A[dm2, dk23] }};
                }
            {% endmacro %}
            
            {% macro copy_B(B1, B) %}
                {% set DN23A, DK23_padded, DN45 = B1.dims %}
                {% set DK23, DN2345 = B.dims %}
                {% set BF, BR, DN23, DN45R = B.gtile(1, DN45) %}
                // B1.dims {{ B1.dims }} B.dims {{ B.dims }}
                {% set dk23 = id_gen.make("dk23") %}
                {% set dn23 = id_gen.make("dn23") %}
                {% set dn45 = id_gen.make("dn45") %}
                #pragma omp for
                FOR({{dk23}}, {{DK23}}) {
                    {% if DN23 != 0 %}
                    FOR({{dn23}}, {{DN23}}) {
                        FOR({{dn45}}, {{DN45}}) {
                            {{ B1[dn23, dk23, dn45] }} = {{ BF[dk23, dn23, dn45] }};
                        }
                    }
                    {% endif %}
                    {% if DN45R != 0 %}
                        FOR({{dn45}}, {{DN45R}}) {
                            {{ B1[DN23, dk23, dn45] }} = {{ BR[dk23, 0, dn45] }};
                        }
                    {% endif %}
                }
            {% endmacro %}
            
            {% macro compute_loop_N_(A1_val, A1_idx, B1, C) %}
                {% set DM3S, DK23 = A1_val.dims %}
                {% set (DK23,) = A1_idx.dims %}
                {% set DN23, DK23, DN45 = B1.dims %}
                {% set DM3, DN2345 = C.dims %}
                {% set CF, CR, DN23, DN45R = C.gtile(1, params['DN45']) %}
                {% set dn23 = id_gen.make("dn23") %}
                {% if DN23 != 0 %}
                FOR({{dn23}}, {{DN23}}) {
                    {{ sparse_dense_gemm_kernel_(A1_val, A1_idx, B1[dn23, '', ''], CF['', dn23, '']) }}
                }
                {% endif %}
                {% if DN45R != 0 %}{
                    {{ sparse_dense_gemm_kernel_(A1_val, A1_idx, B1[DN23, '', ''], CR['', 0, '']) }}
                }{% endif %}
            {% endmacro %}
            
            {% macro compute_loop_(A1_val, A1_idx, B1, C, parallel) %}
                {% set DM2A, DK23 = A1_idx.dims %}
                {% set DM23SA, DK23 = A1_val.dims %}
                {% set DM3S = DM23SA // DM2A %} // never has remainder
                {% set DN23, DK23, DN45 = B1.dims %}
                {% set DM23, DN2345 = C.dims %}
                
                {% set CF, CR, DM2, DM3R = C.gtile(0, params['DM3']) %}
                
                {% set dm2 = id_gen.make("dm2") %}
                {% if DM2 != 0 %}{
                    {% if parallel %}
                    #pragma omp for nowait
                    {% endif %}
                    FOR({{dm2}}, {{DM2}}) {
                        {{ compute_loop_N_(A1_val((DM2A, DM3S), ())[dm2, '', ''], A1_idx[dm2, ''], B1, CF[dm2, '', '']) }}
                    }
                }{% endif %}
                {% if DM3R != 0 %}{
                    {% if parallel %}
                    #pragma omp single nowait
                    {% endif %}
                    {
                        {{ compute_loop_N_(A1_val((DM2A, DM3S), ())[DM2, '', ''], A1_idx[DM2, ''], B1, CR[0, '', '']) }}
                    }
                }{% endif %}
            {% endmacro %}
            
            {% macro compute_head_(A1_val, A1_idx, A_val, A_idx, B1, C) %}
                {% set DM123S, DK23 = A1_val.dims %}
                {% set DM12, DK23 = A1_idx.dims %}
                // A1_val.dims {{ A1_val.dims }}  A1_idx.dims {{ A1_idx.dims }}
                {% set CF, CR, DM1, DM23R = C.gtile(0, params["DM2"] * params["DM3"]) %}
                {% set DM1A = DM1 + (DM23R != 0) %}
                {% set A1_val_exp = A1_val((DM1A, params["DM2"] * params["DM3S"]), ()) %}
                {% set A1_idx_exp = A1_idx((DM1A, params["DM2"]), ()) %}
                {% set A_val_exp = A_val((DM1A, params["DM2"] * params["DM3S"]), ())  %}
                {% set A_idx_exp = A_idx((DM1A, params["DM2"]), ()) %}
                // CF.dims {{ CF.dims }} CR.dims {{ CR.dims }} DM1 {{ DM1 }} DM23R {{ DM23R }}
                {% set dm1 = id_gen.make("dm1") %}
                {% if DM1 != 0 %}{
                    #pragma omp for nowait
                    FOR ({{dm1}}, {{DM1}}) {
                        {{ copy_A(A1_val_exp[dm1, '', ''], A_val_exp[dm1, '', '']) }}
                        {{ copy_A(A1_idx_exp[dm1, '', ''], A_idx_exp[dm1, '', '']) }}
                        {{ compute_loop_(
                            A1_val_exp[dm1, '', ''],
                            A1_idx_exp[dm1, '', ''], 
                            B1,
                            CF[dm1, '', ''],
                            parallel=False) }}
                    }
                }{% endif %}
                {% if DM23R != 0 %}{
                    {% set A1_idx_rem = A1_idx_exp[DM1, '', ''].shorten(((DM12 % params["DM2"]) if (DM12 % params["DM2"]) else (params["DM2"])), None) %}
                    {% set A1_val_rem = A1_val_exp[DM1, '', ''].shorten(A1_idx_rem.dims[0] * params["DM3S"], None) %}
                    {% set A_idx_rem = A_idx_exp[DM1, '', ''].shorten(((DM12 % params["DM2"]) if (DM12 % params["DM2"]) else (params["DM2"])), None) %}
                    {% set A_val_rem = A_val_exp[DM1, '', ''].shorten(A_idx_rem.dims[0] * params["DM3S"], None)  %}
                    #pragma omp single nowait
                    {
                        {{ copy_A(A1_val_rem, A_val_rem) }}
                        {{ copy_A(A1_idx_rem, A_idx_rem) }}
                        {{ compute_loop_(
                            A1_val_rem,
                            A1_idx_rem, 
                            B1,
                            CR[0, '', ''],
                            parallel=False) }}
                    }
                }{% endif %}
            {% endmacro %}
            
            {% macro compute_head_or_tail_(B1, B, A1_val, A1_idx, A_val, A_idx, C, is_head) %}
                {% set DN23A, DK23F, DN45 = B1.dims %}
                {% set DK23, DN2345 = B.dims %}
                {% set B1 = B1.shorten(None, DK23, None) %}
                #pragma omp parallel
                {
                    {{ copy_B(B1, B) }}
                    {% if is_head %}
                        {{ compute_head_(A1_val, A1_idx, A_val, A_idx, B1, C) }}
                    {% else %}
                        {{ compute_loop_(A1_val, A1_idx, B1, C, parallel=True) }}
                    {% endif %}
                }
            {% endmacro %}

            {% macro compute_head_or_tail_2_(B1, B, A1_val, A1_idx, A_val, A_idx, C) %}
                {% set BF, BR, DN1, DN2345R = B.gtile(1, params["DN2345"]) %}
                {% set CF, CR, DN1, DN2345R = C.gtile(1, params["DN2345"]) %}
                {% set dn1 = id_gen.make("dn1") %}
                {% if DN1 == 0 and DN2345R != 0 %}
                {
                    int64_t {{dn1}} = 0;
                    {{ B1.decl() }}
                    {{ compute_head_or_tail_(B1, BR['', 0, ''], A1_val, A1_idx, A_val, A_idx, CR['', 0, ''], is_head=True) }}
                }
                {% else %}
                    {% if DN1 != 0 %}
                    {
                        int64_t {{dn1}} = 0;
                        {{ B1.decl() }}
                        {{ compute_head_or_tail_(B1, BF['', dn1, ''], A1_val, A1_idx, A_val, A_idx, CF['', dn1, ''], is_head=True) }}
                    }
                    for (int64_t {{dn1}} = 1; {{dn1}} < {{DN1}}; {{dn1}}++) {
                        {{ B1.decl() }}
                        {{ compute_head_or_tail_(B1, BF['', dn1, ''], A1_val, A1_idx, A_val, A_idx, CF['', dn1, ''], is_head=False) }}
                    }
                    {% endif %}
                    {% if DN2345R != 0 %}{
                        {{ B1.decl() }}
                        {{ compute_head_or_tail_(B1, BR['', 0, ''], A1_val, A1_idx, A_val, A_idx, CR['', 0, ''], is_head=False) }}
                    }{% endif %}
                {% endif %}
            {% endmacro %}
            
            {% set BF, BR, DK1, DK23R = B.gtile(0, params["DK23"]) %}
            {% set DK1A = DK1 + (DK23R != 0) %}
            
            // BF.dims {{ BF.dims }} BR.dims {{ BR.dims }}
            
            {% set dk1 = id_gen.make("dk1") %}
            {% if DK1 != 0 %}
            FOR({{dk1}}, {{DK1}}) {
                // dk1 main BF.dims {{ BF.dims }}
                {{ A1_val.decl() }}
                {{ A1_idx.decl() }}
                // A1_val.dims {{ A1_val.dims }} A1_idx.dims {{ A1_idx.dims }}
                {{ compute_head_or_tail_2_(B1, BF[dk1, '', ''], A1_val, A1_idx, A_val((), (DK1A, params["DK23"]))['', dk1, ''], A_idx((), (DK1A, params["DK23"]))['', dk1, ''], C) }}
            }
            {% endif %}
            {% if DK23R != 0 %}
            {
                // dk1 remainder BR.dims {{ BR.dims }}
                {{ A1_val.decl() }}
                {{ A1_idx.decl() }}
                {{ compute_head_or_tail_2_(B1, BR[0, '', ''], A1_val, A1_idx, A_val((), (DK1A, params["DK23"]))['', DK1, ''], A_idx((), (DK1A, params["DK23"]))['', DK1, ''], C) }}
            }
            {% endif %}
        }
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    code = template.render({**globals(), **locals()})

    template = jinja2.Template(
        dedent(
            """\
        // WARNING! THIS IS GENERATED FILE! DO NOT EDIT!
        
        #pragma once
        
        #include <stdint.h>
        
        #ifdef __cplusplus
        extern "C" {
        #endif
        
        /*
        Configuration:
        {{ json.dumps(params) }}
        */

        {{ signature }};

        #ifdef __cplusplus
        } // extern "C"
        #endif
    """
        )
    )
    header = template.render({**globals(), **locals()})

    pathlib.Path("generated").mkdir(exist_ok=True)

    with open(f"generated/sparse_dense_{encoded_name}.c", "w") as f:
        f.write(code)
    with open(f"generated/sparse_dense_{encoded_name}.h", "w") as f:
        f.write(header)

    main_template = jinja2.Template(
        dedent(
            """\
        #include "openblas/cblas.h"

        #include "common.hpp"

        #include "generated/sparse_dense_{{ encoded_name }}.h"

        int main() {
            int64_t M = {{ DM }};
            int64_t K = {{ DK }};
            int64_t N = {{ DN }};
            int64_t DK3 = {{ DK3 }};

            std::vector<float> dA = rand_gen(M * K);
            std::vector<float> dB = rand_gen(K * N);

            std::vector<float> dAsv;
            std::vector<int8_t> dAsi;
            std::vector<int16_t> dAsr;
            std::tie(dA, dAsv, dAsi, dAsr) = drop_m_n_non_leading(dA, K, DK3, {{ DM3S }}, {{ DM3 }}, {{ bti_str }});
            
            int K_padded = {{ params["DK_padded"] }};
            int M_padded = {{ params["DM_padded"] }};
            FAIL_CHECK(dA.size() == K_padded * M_padded);
            
            std::vector<float> dA_ref = dA;
            std::vector<float> dB_ref = dB;
            
            int trans_a = {{ 1 * params["trans_a"] }};
            int trans_b = {{ 1 * params["trans_b"] }};
            int trans_c = {{ 1 * params["trans_c"] }};
            if (trans_a) {
                dA = transpose(dA, {{ params["DK_padded"] }});
                dAsv = transpose(dAsv, {{ params["DK_padded"] }});
                dAsi = transpose(dAsi, {{ params["DK_padded"] }});
                dAsr = transpose(dAsr, {{ params["DK_padded"] }});
            }
            if (trans_b) {
                dB = transpose(dB, {{ params["DN"] }});
            }
            
            std::vector<float> dCd(M * N);

            std::vector<float> dCd_ref(M * N);
            
            moment t1, t2;
            
            {
                t1 = timer::now();
                cblas_sgemm(
                    CblasRowMajor, // CBLAS_LAYOUT layout,
                    CblasNoTrans, // CBLAS_TRANSPOSE TransA,
                    CblasNoTrans, // CBLAS_TRANSPOSE TransB,
                    M, // const CBLAS_INDEX M,
                    N, // const CBLAS_INDEX N,
                    K, // const CBLAS_INDEX K,
                    1.0, // const float alpha,
                    dA_ref.data(), // const float *A,
                    K_padded, // const CBLAS_INDEX lda,
                    dB_ref.data(), // const float *B,
                    N, // const CBLAS_INDEX ldb,
                    0.0, // const float beta,
                    dCd_ref.data(), // float *C,
                    N // const CBLAS_INDEX ldc
                );
                t2 = timer::now();
                printf("openblas_dense_dense seconds %.3g ns_per_fma %.3g\\n", seconds(t2 - t1), seconds(t2 - t1) / (M * N * K) * 1e9);
            }
            
            if (trans_c) {
                dCd_ref = transpose(dCd_ref, {{ params["DN"] }});
            }

            {
                std::fill(dCd.begin(), dCd.end(), 0.0f);
                t1 = timer::now();
                sparse_dense_{{ encoded_name }}(
                    dAsv.data(), // const float *A,
                    dAsr.data(),
                    dB.data(), // const float *B,
                    dCd.data() // float *C,
                );
                t2 = timer::now();
                printf("my_sparse_dense seconds %.3g ns_per_fma %.3g sparsity %.2f pattern %d:%d\\n", seconds(t2 - t1), seconds(t2 - t1) / (M * N * K * {{ DM3S / DM3 }}) * 1e9, {{ 1.0 - DM3S / DM3 }}, {{ DM3S }}, {{ DM3 }});
                CHECK(vector_allclose(dCd, dCd_ref));
            }
        }
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    main = main_template.render({**globals(), **locals()})

    with open(f"generated/sparse_dense_{encoded_name}_main.cpp", "w") as f:
        f.write(main)

    return encoded_name


def generate_grouped_m_n_converter(DK3, DM3, DM3S):
    bti = make_blk_to_idx_list(DM3S, DM3)
    bti_str = (
        "{"
        + ", ".join(
            "{" + ", ".join(str(idx) for idx in idx_tuple) + "}" for idx_tuple in bti
        )
        + "}"
    )

    converter_template = jinja2.Template(
        dedent(
            """\
        #include "common.hpp"

        extern "C" {
        void dense_to_sparse_GRP{{DK3}}_M{{DM3S}}_N{{DM3}}(float* sparse_vals, int16_t* sparse_indices, float* dense_vals, int64_t DM, int64_t DK, float* sparsified_dense_vals) {
            std::vector<float> dA(dense_vals, dense_vals + DM * DK);
            std::vector<float> sdA;
            std::vector<float> dAsv;
            std::vector<int8_t> dAsi;
            std::vector<int16_t> dAsr;
            std::tie(sdA, dAsv, dAsi, dAsr) = drop_m_n_non_leading(dA, DK, {{ DK3 }}, {{ DM3S }}, {{ DM3 }}, {{ bti_str }});
            std::copy(dAsv.begin(), dAsv.end(), sparse_vals);
            std::copy(dAsr.begin(), dAsr.end(), sparse_indices);
            if (sparsified_dense_vals) {
                std::copy(sdA.begin(), sdA.end(), sparsified_dense_vals);
            }
        }
        }
        
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    with open(
        f"generated/sparse_dense_grouped{DK3}_m{DM3S}_n{DM3}_converter.cpp", "w"
    ) as f:
        f.write(converter_template.render({**globals(), **locals()}))


def benchmark_parametrized_sparse_dense(params):
    encoded_name = generate_parametrized_sparse_dense(params)

    print("Compilation started...")
    subprocess.run(
        f"gcc -march=native -g -O3 -fopenmp -lpthread generated/sparse_dense_{encoded_name}.c -c -o generated/sparse_dense_{encoded_name}.o".split(),
        check=True,
    )
    subprocess.run(
        f"g++ -march=native -g -O3 -fopenmp -lpthread -I. -I./OpenBLAS/install/include generated/sparse_dense_{encoded_name}_main.cpp ./OpenBLAS/install/lib/libopenblas.a generated/sparse_dense_{encoded_name}.o -o generated/sparse_dense_{encoded_name}_main".split(),
        check=True,
    )
    print("Compilation completed")
    subprocess.run(f"generated/sparse_dense_{encoded_name}_main", check=True)


def derive_params(params_base):
    params = dict(params_base)

    # A is always sparse, B is always dense
    params["DMI_padded"] = math.ceil(params["DM"] / params["DM3"])
    params["DMS_padded"] = params["DMI_padded"] * params["DM3S"]
    params["DM_padded"] = params["DMI_padded"] * params["DM3"]
    params["DK2"] = (
        math.factorial(params["DM3"])
        // math.factorial(params["DM3S"])
        // math.factorial(params["DM3"] - params["DM3S"])
    )
    params["DK_padded"] = math.ceil(params["DK"] / (params["DK2"] * params["DK3"])) * (
        params["DK2"] * params["DK3"]
    )

    if params["trans_a"]:
        params["SAM"] = 1
        params["SAK"] = params["DM_padded"]
        params["SAMS"] = 1
        params["SAKS"] = params["DMS_padded"]
        params["SAMI"] = 1
        params["SAKI"] = params["DMI_padded"]
    else:
        params["SAM"] = params["DK_padded"]
        params["SAK"] = 1
        params["SAMS"] = params["DK_padded"]
        params["SAKS"] = 1
        params["SAMI"] = params["DK_padded"]
        params["SAKI"] = 1

    if params["trans_b"]:
        params["SBK"] = 1
        params["SBN"] = params["DK"]
    else:
        params["SBK"] = params["DN"]
        params["SBN"] = 1

    if params["trans_c"]:
        params["SCM"] = 1
        params["SCN"] = params["DM"]
    else:
        params["SCM"] = params["DN"]
        params["SCN"] = 1

    return params


def test_baseline_sparse_dense():
    for trans_c in (True, False):
        for trans_a in (False, True):
            for trans_b in (False, True):
                params = derive_params(
                    {
                        "trans_a": trans_a,
                        "trans_b": trans_b,
                        "trans_c": trans_c,
                        "DM": 4000,
                        "DK": 3000,
                        "DN": 2000,
                        "DK3": 16,
                        "DM3": 6,
                        "DM3S": 3,
                        "DM2": 3,
                        "DN5": 8,
                        "DN4": 4,
                        "DN3": 1,
                        "DN2": 16,
                    }
                )

                benchmark_parametrized_sparse_dense(params)


def baseline_sparse_dense():
    params = derive_params(
        {
            "trans_a": False,
            "trans_b": False,
            "trans_c": False,
            "DM": 4000,
            "DK": 3000,
            "DN": 2000,
            "DK3": 16,
            "DM3": 6,
            "DM3S": 3,
            "DM2": 3,
            "DN5": 8,
            "DN4": 4,
            "DN3": 1,
            "DN2": 16,
        }
    )

    benchmark_parametrized_sparse_dense(params)


def baseline_dense_dense():
    params = {
        "target_M": 2048,
        "target_N": 2048,
        "target_K": 2048,
        "DK3": 8,
        "DK2": 40,
        "DM3": 12,
        "DM2": 2,
        "DN5": 8,
        "DN4": 1,
        "DN3": 20,
        "DN2": 1,
        "loop_order_copy_b": list(range(6)),
        "loop_order_copy_a": list(range(4)),
        "loop_order_inner": list(range(3)),
        "B1_layout": list(range(6)),
        "A1_layout": list(range(4)),
    }

    benchmark_parametrized_dense_dense(params)


def test_sparse_dense():
    DK3 = 16
    DM3 = 6
    DM3S = 3
    DM2 = 3
    DM1 = 5
    DN5 = 8
    DN4 = 4
    DN3 = 1
    DN2 = 16
    DN1 = 13
    DK2 = math.factorial(DM3) // math.factorial(DM3S) // math.factorial(DM3 - DM3S)
    DK1 = 7

    DMs = [
        1,
        DM3 - 1,
        DM3,
        DM2 * DM3 - 1,
        DM2 * DM3,
        DM1 * DM2 * DM3 - 1,
        DM1 * DM2 * DM3,
    ]
    DNs = [
        1,
        DN5 - 1,
        DN5,
        DN4 * DN5 - 1,
        DN4 * DN5,
        DN2 * DN3 * DN4 * DN5 - 1,
        DN2 * DN3 * DN4 * DN5,
        DN1 * DN2 * DN3 * DN4 * DN5 - 1,
        DN1 * DN2 * DN3 * DN4 * DN5,
    ]
    DKs = [
        1,
        DK3 - 1,
        DK3,
        DK2 * DK3 - 1,
        DK2 * DK3,
        DK1 * DK2 * DK3 - 1,
        DK1 * DK2 * DK3,
    ]

    MNK = []
    for DM in DMs:
        MNK.append((DM, DNs[0], DKs[0]))
        # MNK.append((DM, DNs[-1], DKs[-1]))
    for DN in DNs:
        MNK.append((DMs[0], DN, DKs[0]))
        # MNK.append((DMs[-1], DN, DKs[-1]))
    for DK in DKs:
        MNK.append((DMs[0], DNs[0], DK))
        # MNK.append((DMs[-1], DNs[-1], DK))

    for DM, DN, DK in MNK:
        params = {
            "DM": DM,
            "DK": DK,
            "DN": DN,
            "DK3": DK3,
            "DM3": DM3,
            "DM3S": DM3S,
            "DM2": DM2,
            "DN5": DN5,
            "DN4": DN4,
            "DN3": DN3,
            "DN2": DN2,
            "trans_a": False,
            "trans_b": False,
            "trans_c": False,
        }
        benchmark_parametrized_sparse_dense(params)


def get_tile_used_memory(M, K, N):
    return (M * K + K * N + M * N) * 4


def test_bert_sizes1():
    M = 768
    K = 3072
    N = 4096

    params = {
        "target_M": M,
        "target_N": N,
        "DK": K,
        "DK3": 8,
        "DK2": 40,
        "DM3": 12,
        "DM2": 2,
        "DN5": 8,
        "DN4": 1,
        "DN3": 20,
        "DN2": 1,
        "loop_order_copy_b": list(range(6)),
        "loop_order_copy_a": list(range(4)),
        "loop_order_inner": list(range(3)),
        "B1_layout": list(range(6)),
        "A1_layout": list(range(4)),
    }
    benchmark_parametrized_dense_dense(params)

    return

    acc_size = 4

    n = 3
    m = 6

    chunks = fact(m) // fact(n) // fact(m - n)  # DK2

    g = 16

    tile_b_size = 16
    tile_a_size = 4

    params = {
        "DM": 768,
        "DK": 3072,
        "DN": 4096,
        "DK3": g,  # g in n:m:g
        "DM3": m,  # m in n:m:g
        "DM3S": n,  # n in n:m:g
        "DM2": tile_a_size,  # tile size A
        "DN5": 8,  # vector size
        "DN4": acc_size,  # accumulator DN4 x DM3S
        "DN3": 1,  # unused tile dim
        "DN2": tile_b_size,  # tile size B
        "trans_a": False,
        "trans_b": False,
        "trans_c": False,
    }

    changes_per_tile = [
        ("N", 8),  # DN5
        ("M", n),  # DM3S
        ("N", acc_size),  # DN4
        ("K", g),  #  DK3
        ("K", chunks),  # DK2
        ("N", tile_b_size),  # DN23
        ("M", tile_a_size),  # DM2
        ("M", M // (tile_a_size * n)),  # DM1
        ("N", N // (tile_b_size * acc_size * 8)),  # DN1
        ("K", K // (chunks * g)),  # DK1
    ]
    sizes_per_tile = []
    for i, e in enumerate(changes_per_tile):
        sizes = {"N": 1, "K": 1, "M": 1}
        for j in range(i):
            dim, size = changes_per_tile[j]
            sizes[dim] *= size
        sizes_per_tile.append(get_tile_used_memory(**sizes))

    print(sizes_per_tile)

    cores = 8
    # sudo dmidecode -t cache / lscpu
    total_l1 = 256 * 1024  # 32 * 1024 per core
    total_l2 = 1024 * 1024  # 256 * 1024 per pair of cores
    total_l3 = 8182 * 1024
    l1 = total_l1 // cores
    l2 = total_l2 // cores
    l3 = total_l3 // cores
    print(f"acc_size {acc_size} * n {n} = {acc_size * n} < 16")
    assert acc_size * n <= 16  # try to reduce acc_size
    print(f"sizes_per_tile[5] {sizes_per_tile[5]} < {l1} (g)")
    # assert sizes_per_tile[5] <= l1 # L1d cache: 32K, try to reduce group size g
    print(f"sizes_per_tile[6] {sizes_per_tile[6]} < {l2} (tile_b_size)")
    # assert sizes_per_tile[6] <= l2 # L2 cache: 256K, try to reduce tile_b_size
    print(f"sizes_per_tile[7] {sizes_per_tile[7]} < {l3} (tile_a_size)")
    # assert sizes_per_tile[7] <= l3 # L3 cache: 8182K, try to reduce tile_a_size

    benchmark_parametrized_sparse_dense(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sparse-baseline", action=argparse.BooleanOptionalAction)
    parser.add_argument("--run-dense-baseline", action=argparse.BooleanOptionalAction)
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.run_sparse_baseline:
        baseline_sparse_dense()
    if args.run_dense_baseline:
        baseline_dense_dense()
    if args.run_tests:
        test_sparse_dense()
        test_baseline_sparse_dense()
    test_bert_sizes1()
