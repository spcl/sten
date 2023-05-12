from dataclasses import dataclass
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
from typing import Tuple, List, Union
import argparse
from .grouped_nm_tensor import GroupedNMTensor, is_correct_nm, make_n_m, int_ceil
import dace
from dace import nodes
import jinja2
from textwrap import dedent
from .matmul_generator import make_load_mask


def generate_sparse_dense_microkernel_avx2(
    ptr_a_val,
    ptr_a_idx,
    ptr_b,
    ptr_c,
    M,
    K,
    Nblocks,
    SAMS,
    SAKS,
    SAMI,
    SAKI,
    SBK,
    SCM,
    SCB,
    SCV,
    SBB,
    SBV,
    n,
    m,
    g,
    num_vectors_b,
    v,
):
    DK2 = fact(m) // fact(n) // fact(m - n)
    DK3 = g
    DM3 = m
    DM3S = n
    DN4 = num_vectors_b  # accumulator tile size
    DN5 = v  # vector size

    bti = make_n_m(n, m)

    DM3S_padded = int_ceil(M, DM3) * DM3S
    DK23_padded = int_ceil(K, DK2 * DK3) * DK2 * DK3
    # DN5R = N % DN5
    DN5R = 0

    assert M <= DM3
    assert K <= DK2 * DK3
    # assert N <= DN4 * DN5
    assert SCV == 1
    assert SBV == 1

    template = jinja2.Template(
        dedent(
            """\
        // SAMS, SAKS, SAMI, SAKI, SBK, SCM, SCB, SCV, SBB, SBV = {{SAMS, SAKS, SAMI, SAKI, SBK, SCM, SCB, SCV, SBB, SBV}}
        // init accumulators
        {% for dm3s in range(DM3S_padded) %}
        {% for dn4 in range(DN4) %}
        __m256 acc_{{ dm3s }}_{{ dn4 }} = _mm256_setzero_ps();
        {% endfor %}
        {% endfor %}
        {% if DN5R != 0 %}__m256i load_mask = _mm256_setr_epi32({{ make_load_mask(DN5R, DN5) }});{% endif %}
        {% for dk2 in range(DK2) %}{
            {% set si = bti[dk2] %}
            {% set in_remainder_DK2 = (DK23_padded != K) %}
            // in_remainder_DK2 = {{ in_remainder_DK2 }} dk2 {{ dk2 }} DK2 {{ DK2 }} DK23_padded {{ DK23_padded }} K {{ K }}
            FOR(dk3, {{DK3}}) {
                int dk23 = {{ dk2 }} * {{ DK3 }} + dk3;
                int16_t dk23_in_B = {{ ptr_a_idx }}[dk23 * {{ SAKI }}];
                //printf("A LOAD [offset from A = %d, dk23 = %d, dk3=%d, DK3=%d, SAKI = %d]\\n", &({{ ptr_a_idx }}[dk23 * {{ SAKI }}]) - A_idx, dk23, dk3, {{DK3}} , {{ SAKI }});
                //printf("dk23 in B = %d\\n", (int)dk23_in_B);
                {% if in_remainder_DK2 %}
                    // in dk2 remainder
                    if ((dk23_in_B < 0) || (dk23_in_B >= {{K}})) break;
                {% endif %}
                // SBK = {{SBK}} 
                float* B2 = &{{ ptr_b }}[dk23_in_B * {{ SBK }}];
                //printf("B2 offset [offset from B = %d, dk23_in_b = %d, SBK = %d]\\n", B2 - B, dk23_in_B, {{SBK}});
                {% for dm3s in range(DM3S_padded) %}
                    __m256 va_{{ dm3s }} = _mm256_broadcast_ss(&{{ ptr_a_val }}[{{ dm3s }} * {{ SAMS }} + dk23 * {{ SAKS }}]);
                {% endfor %}
                {% for dn4 in range(DN4) %}{
                    {% set in_remainder_DN5 = (dn4 == DN4 - 1) and (DN5R != 0) %}
                    {% if not in_remainder_DN5 %}
                        //printf("B2 LOAD [offset from B = %d, dn4 = %d, SBB = %d]\\n", &B2[{{ dn4 }} * {{ SBB }}] - B, {{ dn4}} , {{ SBB }});
                        __m256 vb_{{ dn4 }} = _mm256_loadu_ps(&B2[{{dn4}} * {{SBB}}]);
                    {% else %}
                        __m256 vb_{{ dn4 }} = _mm256_maskload_ps(&B2[{{ dn4 }} * {{ SBB }}], load_mask);
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
                           float* c_addr = &{{ ptr_c }}[{{ dm3 }} * {{ SCM }} + {{ dn4 }} * {{ SCB }}];
                           {% if not in_remainder_DN5 %}
                               _mm256_storeu_ps(c_addr, _mm256_add_ps(_mm256_loadu_ps(c_addr), acc_{{ dm3s }}_{{ dn4 }}));
                           {% else %}
                               _mm256_maskstore_ps(c_addr, load_mask, _mm256_add_ps(_mm256_maskload_ps(c_addr, load_mask), acc_{{ dm3s }}_{{ dn4 }}));
                           {% endif %}
                        {% endif %}
                        {% if (dk2 != DK2 - 1) %}
                            acc_{{ dm3s }}_{{ dn4 }} = _mm256_setzero_ps();
                        {% endif %}
                    }{% endfor %}
                {% endif %}
            {% endfor %}
        }{% endfor %}
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    code = template.render({**globals(), **locals()})

    return code


def generate_sparse_dense_microkernel_avx512(
    ptr_a_val,
    ptr_a_idx,
    ptr_b,
    ptr_c,
    M,
    K,
    Nblocks,
    SAMS,
    SAKS,
    SAMI,
    SAKI,
    SBK,
    SCM,
    SCB,
    SCV,
    SBB,
    SBV,
    n,
    m,
    g,
    num_vectors_b,
    v,
):
    DK2 = fact(m) // fact(n) // fact(m - n)
    DK3 = g
    DM3 = m
    DM3S = n
    DN4 = num_vectors_b  # accumulator tile size
    DN5 = v  # vector size

    bti = make_n_m(n, m)

    DM3S_padded = int_ceil(M, DM3) * DM3S
    DK23_padded = int_ceil(K, DK2 * DK3) * DK2 * DK3
    # DN5R = N % DN5
    DN5R = 0

    assert M <= DM3
    assert K <= DK2 * DK3
    # assert N <= DN4 * DN5
    assert SCV == 1
    assert SBV == 1

    template = jinja2.Template(
        dedent(
            """\
        // SAMS, SAKS, SAMI, SAKI, SBK, SCM, SCB, SCV, SBB, SBV = {{SAMS, SAKS, SAMI, SAKI, SBK, SCM, SCB, SCV, SBB, SBV}}
        // init accumulators
        {% for dm3s in range(DM3S_padded) %}
        {% for dn4 in range(DN4) %}
        __m512 acc_{{ dm3s }}_{{ dn4 }} = _mm512_setzero_ps();
        {% endfor %}
        {% endfor %} 
        {% if DN5R != 0 %}__mmask16 load_mask = {{ 2 ** DN5R - 1 }};{% endif %}
        {% for dk2 in range(DK2) %}{
            {% set si = bti[dk2] %}
            {% set in_remainder_DK2 = (DK23_padded != K) %}
            // in_remainder_DK2 = {{ in_remainder_DK2 }} dk2 {{ dk2 }} DK2 {{ DK2 }} DK23_padded {{ DK23_padded }} K {{ K }}
            FOR(dk3, {{DK3}}) {
                int dk23 = {{ dk2 }} * {{ DK3 }} + dk3;
                int16_t dk23_in_B = {{ ptr_a_idx }}[dk23 * {{ SAKI }}];
                //printf("A LOAD [offset from A = %d, dk23 = %d, dk3=%d, DK3=%d, SAKI = %d]\\n", &({{ ptr_a_idx }}[dk23 * {{ SAKI }}]) - A_idx, dk23, dk3, {{DK3}} , {{ SAKI }});
                //printf("dk23 in B = %d\\n", (int)dk23_in_B);
                {% if in_remainder_DK2 %}
                    // in dk2 remainder
                    if ((dk23_in_B < 0) || (dk23_in_B >= {{K}})) break;
                {% endif %}
                // SBK = {{SBK}}  
                float* B2 = &{{ ptr_b }}[dk23_in_B * {{ SBK }}];
                //printf("B2 offset [offset from B = %d, dk23_in_b = %d, SBK = %d]\\n", B2 - B, dk23_in_B, {{SBK}});
                {% for dm3s in range(DM3S_padded) %}
                    __m512 va_{{ dm3s }} = _mm512_set1_ps({{ ptr_a_val }}[{{ dm3s }} * {{ SAMS }} + dk23 * {{ SAKS }}]);
                {% endfor %}
                {% for dn4 in range(DN4) %}{
                    {% set in_remainder_DN5 = (dn4 == DN4 - 1) and (DN5R != 0) %}
                    {% if not in_remainder_DN5 %}
                        //printf("B2 LOAD [offset from B = %d, dn4 = %d, SBB = %d]\\n", &B2[{{ dn4 }} * {{ SBB }}] - B, {{ dn4}} , {{ SBB }});
                        __m512 vb_{{ dn4 }} = _mm512_loadu_ps(&B2[{{dn4}} * {{SBB}}]);
                    {% else %}
                        __m512 vb_{{ dn4 }} = _mm512_maskz_load_ps(load_mask, &B2[{{ dn4 }} * {{ SBB }}]);
                    {% endif %}
                    {% for dm3s in range(DM3S_padded) %}
                        acc_{{ dm3s }}_{{ dn4 }} = _mm512_fmadd_ps(va_{{ dm3s }}, vb_{{ dn4 }}, acc_{{ dm3s }}_{{ dn4 }});
                    {% endfor %}
                }{% endfor %}
            }
            {% for dm3s in range(DM3S_padded) %}
                {% if (dk2 == DK2 - 1) or (bti[dk2][dm3s] != bti[dk2+1][dm3s]) %}
                    {% for dn4 in range(DN4) %}{
                        {% set in_remainder_DN5 = (dn4 == DN4 - 1) and (DN5R != 0) %}
                        {% set dm3 = si[dm3s] %}
                        {% if dm3 < DM3 %} // cut out remainder DM3
                           float* c_addr = &{{ ptr_c }}[{{ dm3 }} * {{ SCM }} + {{ dn4 }} * {{ SCB }}];
                           {% if not in_remainder_DN5 %}
                               _mm512_storeu_ps(c_addr, _mm512_add_ps(_mm512_loadu_ps(c_addr), acc_{{ dm3s }}_{{ dn4 }}));
                           {% else %}
                               _mm512_mask_store_ps(c_addr, load_mask, _mm512_add_ps(_mm512_maskz_load_ps(load_mask, c_addr), acc_{{ dm3s }}_{{ dn4 }}));
                           {% endif %}
                        {% endif %}
                        {% if (dk2 != DK2 - 1) %}
                            acc_{{ dm3s }}_{{ dn4 }} = _mm512_setzero_ps();
                        {% endif %}
                    }{% endfor %}
                {% endif %}
            {% endfor %}
        }{% endfor %}
    """
        ),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    code = template.render({**globals(), **locals()})

    return code


@dataclass
class Loop:
    dim: str = "m"
    sequential: bool = False
    size: int = 1  # Relative tile size
    buffer_a: bool = False
    buffer_b: bool = False
    buffer_c: bool = False


def generate_configuration(
    n: int,
    m: int,
    g: int,
    M: int,
    N: int,
    K: int,
    B_trans: bool,
    C_trans: bool,
    *,
    vector_size: int = 8,
    num_vectors_b: int = 4,  # Microkernel parameters
    loops: Union[List[Loop], List[List[Loop]]] = [
        Loop("k", sequential=True),
        Loop("m"),
        Loop("n"),
    ],
    local_c: bool = False,
) -> dace.SDFG:
    if vector_size not in (8, 16):
        raise ValueError("vector_size must be 8 or 16")

    num_vector_registers = vector_size * 2  # 16 for AVX2 and AVX512, 32 for AVX512_VL

    assert num_vectors_b >= 1 and num_vectors_b * n < num_vector_registers

    # Ensure integers are 32-bit by default
    dace.Config.set("compiler", "default_data_types", value="C")
    dace.Config.set("compiler.allow_view_arguments", value=True)

    # Create a new SDFG
    sdfg = dace.SDFG("gnm_mult")
    state = sdfg.add_state()

    # Tile sizes
    DK2 = math.factorial(m) // math.factorial(n) // math.factorial(m - n)
    DK2_g = DK2 * g
    DMI_padded = math.ceil(M / m)
    DKI_padded = math.ceil(K / DK2_g)
    Nblocks = N // (num_vectors_b * vector_size)

    # Create C array
    C_desc = dace.float32[DMI_padded, m, Nblocks, num_vectors_b, vector_size]
    if C_trans:
        C_desc.set_strides_from_layout(2, 3, 4, 0, 1)
    sdfg.add_datadesc("C", C_desc)

    # Initialize C to zero
    if not local_c:
        state.add_mapped_tasklet(
            "init",
            dict(
                i=f"0:{DMI_padded}",
                j=f"0:{m}",
                k=f"0:{Nblocks}",
                l=f"0:{num_vectors_b}",
                z=f"0:{vector_size}",
            ),
            {},
            "c = 0",
            {"c": dace.Memlet("C[i, j, k, l, z]")},
            external_edges=True,
        )

    if isinstance(loops[0], Loop):
        all_loops: List[List[Loop]] = [loops] * 8
    else:
        # Specialized loops for every remainder
        all_loops: List[List[Loop]] = loops

    num_tiles_k = DKI_padded
    if K % DK2_g != 0:
        num_tiles_k -= 1
        k_remainder = True
    else:
        k_remainder = False

    # Create microkernel states for each remainder configuration
    for loops, (rem_m, rem_n, rem_k) in zip(
        all_loops,
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ],
    ):
        if rem_n == 1:
            continue
        if rem_k == 1 and k_remainder is False:
            continue
        state = sdfg.add_state_after(state, f"microkernel_{rem_m}_{rem_n}_{rem_k}")

        # Skip remainder loops
        loops = [loop for loop in loops if loop.dim != "m" or rem_m == 0]
        loops = [loop for loop in loops if loop.dim != "n" or rem_n == 0]
        loops = [loop for loop in loops if loop.dim != "k" or rem_k == 0]

        generate_single_kernel(
            sdfg,
            state,
            n,
            m,
            g,
            M,
            N,
            K,
            B_trans,
            C_trans,
            vector_size,
            num_vectors_b,
            loops,
            local_c,
            rem_m,
            rem_n,
            rem_k,
            num_tiles_k,
        )

    sdfg.append_global_code(
        """
#include <immintrin.h>
#define FOR(i, n) for (int i = 0; i < n; ++i)
"""
    )
    sdfg.openmp_sections = False

    # sdfg.simplify()

    return sdfg


def generate_single_kernel(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    n: int,
    m: int,
    g: int,
    M: int,
    N: int,
    K: int,
    B_trans: bool,
    C_trans: bool,
    v: int,
    num_vectors_b: int,
    loops: List[Loop],
    local_c: bool,
    rem_m: bool,
    rem_n: bool,
    rem_k: bool,
    num_tiles_k: int,
):
    kernel_generator = (
        generate_sparse_dense_microkernel_avx2
        if v == 8
        else generate_sparse_dense_microkernel_avx512
    )
    assert N % v == 0
    assert N % (num_vectors_b * v) == 0
    assert rem_n == 0

    # Number of permutations
    DK2 = math.factorial(m) // math.factorial(n) // math.factorial(m - n)
    DK2_g = DK2 * g

    # Sizes
    DMI_padded = math.ceil(M / m)
    DKI_padded = math.ceil(K / DK2_g)
    Nblocks = N // (num_vectors_b * v)
    map_dims = {"m": DMI_padded, "n": Nblocks, "k": num_tiles_k}

    # Tile sizes
    tile_M = m
    tile_K = DK2_g

    # Data descriptors
    A_val_desc = dace.float32[DKI_padded, DMI_padded, DK2, g, n]
    A_idx_desc = dace.int16[DKI_padded, DMI_padded, DK2, g, 1]
    B_desc = dace.float32[K, N]
    C_desc = dace.float32[DMI_padded, m, Nblocks, num_vectors_b, v]
    if B_trans:
        B_desc.set_strides_from_layout(1, 0)
    if C_trans:
        C_desc.set_strides_from_layout(2, 3, 4, 0, 1)

    if "A_val" not in sdfg.arrays:
        sdfg.add_datadesc("A_val", A_val_desc)
        sdfg.add_datadesc("A_idx", A_idx_desc)
        sdfg.add_datadesc("B", B_desc)

    # Add loops to graph
    map_entries: List[nodes.MapEntry] = []
    map_exits: List[nodes.MapExit] = []
    for loop in loops:
        map_entry, map_exit = state.add_map(
            f"{loop.dim}_map",
            {f"block_{loop.dim}": f"0:{map_dims[loop.dim]}"},
            schedule=(
                dace.ScheduleType.Sequential
                if loop.sequential
                else dace.ScheduleType.CPU_Multicore
            ),
        )
        map_entries.append(map_entry)
        map_exits.append(map_exit)

    # Get innermost local storage strides
    SBV, SBB, SBK = 1, v, B_desc.strides[0]
    SCM, SCB, SCV = C_desc.strides[-4], C_desc.strides[-2], C_desc.strides[-1]

    # Generate tasklet code with innermost local storage strides
    autogenerated_code = kernel_generator(
        ptr_a_val="ptr_a_val",
        ptr_a_idx="ptr_a_idx",
        ptr_b="ptr_b",
        ptr_c="ptr_c",
        M=tile_M if (rem_m == 0) else (M % tile_M),
        K=tile_K if (rem_k == 0) else (K % tile_K),
        SAMS=A_val_desc.strides[-1],
        SAKS=A_val_desc.strides[-2],
        SAMI=A_idx_desc.strides[-1],
        SAKI=A_idx_desc.strides[-2],
        SBV=SBV,
        SBB=SBB,
        SBK=SBK,
        SCM=SCM,
        SCB=SCB,
        SCV=SCV,
        n=n,
        m=m,
        g=g,
        Nblocks=Nblocks,
        num_vectors_b=num_vectors_b,
        v=v,
    )

    a_index_k = "block_k" if (rem_k == 0) else f"{DKI_padded-1}"
    a_index_m = "block_m" if (rem_m == 0) else f"{DMI_padded-1}"
    b_index_k = (
        f"block_k*{DK2_g}:(block_k+1)*{DK2_g}"
        if (rem_k == 0)
        else f"{num_tiles_k*DK2*g}:{K}"
    )
    b_index_n = f"block_n*{num_vectors_b*v}:(block_n+1)*{num_vectors_b*v}"

    # Add tasklet
    tasklet = state.add_tasklet(
        "kernel",
        {"ptr_a_val", "ptr_a_idx", "ptr_b", "ptr_cin"},
        {"ptr_c"},
        autogenerated_code,
        language=dace.Language.CPP,
    )

    # Add access nodes
    a_val = state.add_read("A_val")
    a_idx = state.add_read("A_idx")
    b = state.add_read("B")
    cin = state.add_read("C")
    cout = state.add_write("C")

    # Add memlet paths
    state.add_memlet_path(
        a_val,
        *map_entries,
        tasklet,
        dst_conn="ptr_a_val",
        memlet=dace.Memlet(f"A_val[{a_index_k}, {a_index_m}, 0:{DK2}, 0:{g}, 0:{n}]"),
    )
    state.add_memlet_path(
        a_idx,
        *map_entries,
        tasklet,
        dst_conn="ptr_a_idx",
        memlet=dace.Memlet(f"A_idx[{a_index_k}, {a_index_m}, 0:{DK2}, 0:{g}, 0:1]"),
    )
    state.add_memlet_path(
        b,
        *map_entries,
        tasklet,
        dst_conn="ptr_b",
        memlet=dace.Memlet(f"B[{b_index_k}, {b_index_n}]"),
    )
    state.add_memlet_path(
        cin,
        *map_entries,
        tasklet,
        dst_conn="ptr_cin",
        memlet=dace.Memlet(f"C[{a_index_m}, 0:{m}, block_n, 0:{num_vectors_b}, 0:{v}]"),
    )
    state.add_memlet_path(
        tasklet,
        *map_exits[::-1],
        cout,
        src_conn="ptr_c",
        memlet=dace.Memlet(f"C[{a_index_m}, 0:{m}, block_n, 0:{num_vectors_b}, 0:{v}]"),
    )


def nmg_mult(
    outshape: Tuple[int],
    m: int,
    n: int,
    g: int,
    transpose_b: bool,
    transpose_c: bool,
    tile: int,
    tile_2: int,
    local_b: bool,
    local_c: bool,
    name: str = None,
    kernel: str = "avx2",
):
    assert kernel in ("avx2", "avx512")
    kernel_generator = (
        generate_sparse_dense_microkernel_avx2
        if kernel == "avx2"
        else generate_sparse_dense_microkernel_avx512
    )

    import dace

    # Ensure integers are 32-bit by default
    dace.Config.set("compiler", "default_data_types", value="C")
    dace.Config.set("compiler.allow_view_arguments", value=True)

    M, K, N = outshape
    # Number of permutations
    DK2 = math.factorial(m) // math.factorial(n) // math.factorial(m - n)
    DK2_g = DK2 * g
    v = 8 if kernel == "avx2" else 16
    num_vectors_b = 4
    assert N % v == 0
    assert N % (num_vectors_b * v) == 0

    DMI_padded = math.ceil(M / m)
    DM_padded = DMI_padded * m
    DKI_padded = math.ceil(K / DK2_g)
    DK_padded = DKI_padded * DK2_g

    Nblocks = N // (num_vectors_b * v)

    A_val_desc = dace.float32[DKI_padded, DMI_padded, DK2, g, n]
    A_idx_desc = dace.int16[DKI_padded, DMI_padded, DK2, g, 1]
    B_desc = dace.float32[DKI_padded, DK2, g, Nblocks, num_vectors_b, v]
    C_desc = dace.float32[DMI_padded, m, Nblocks, num_vectors_b, v]

    local_B_desc = dace.float32[DK2, g, num_vectors_b, v]
    local_C_desc = dace.float32[DMI_padded, m, num_vectors_b, v]

    # Change strides accordingly
    if transpose_b:
        B_desc.set_strides_from_layout(2, 1, 0, 5, 4, 3)
    if transpose_c:
        C_desc.set_strides_from_layout(1, 0, 4, 3, 2)

    # Handle strides in local storage
    if local_b:
        SBV, SBB, SBK = (
            local_B_desc.strides[-1],
            local_B_desc.strides[-2],
            local_B_desc.strides[-3],
        )
    else:
        SBV, SBB, SBK = B_desc.strides[-1], B_desc.strides[-2], B_desc.strides[-4]

    if local_c:
        SCM, SCB, SCV = (
            local_C_desc.strides[-3],
            local_C_desc.strides[-2],
            local_C_desc.strides[-1],
        )
    else:
        SCM, SCB, SCV = C_desc.strides[-4], C_desc.strides[-2], C_desc.strides[-1]

    AUTOGENERATED_CODE = [[[";", ";"], [";", ";"]], [[";", ";"], [";", ";"]]]

    tile_M = m
    tile_K = DK2_g
    tile_N = v * num_vectors_b

    for mrem in range(2):
        for nrem in range(2):
            for krem in range(2):
                if mrem and (M % tile_M) < 0:
                    raise ValueError("Invalid tile_M")
                if mrem and (M % tile_M) == 0:
                    continue
                if nrem and (N % tile_N) < 0:
                    raise ValueError("Invalid tile_N")
                if nrem and (N % tile_N) == 0:
                    continue
                if krem and (K % tile_K) < 0:
                    raise ValueError("Invalid tile_K")
                if krem and (K % tile_K) == 0:
                    continue

                AUTOGENERATED_CODE[mrem][nrem][krem] = kernel_generator(
                    ptr_a_val="ptr_a_val",
                    ptr_a_idx="ptr_a_idx",
                    ptr_b="ptr_b",
                    ptr_c="ptr_c",
                    M=tile_M if (mrem == 0) else (M % tile_M),
                    K=tile_K if (krem == 0) else (K % tile_K),
                    SAMS=A_val_desc.strides[-1],
                    SAKS=A_val_desc.strides[-2],
                    SAMI=A_idx_desc.strides[-1],
                    SAKI=A_idx_desc.strides[-2],
                    SBV=SBV,
                    SBB=SBB,
                    SBK=SBK,
                    SCM=SCM,
                    SCB=SCB,
                    SCV=SCV,
                    n=n,
                    m=m,
                    g=g,
                    Nblocks=Nblocks,
                    num_vectors_b=num_vectors_b,
                    v=v,
                )
    AUTOGENERATED_CODE_0_0_0 = "/* XXX 000 XXX */\n" + AUTOGENERATED_CODE[0][0][0]
    AUTOGENERATED_CODE_0_0_1 = "/* XXX 00K XXX */\n" + AUTOGENERATED_CODE[0][0][1]
    AUTOGENERATED_CODE_0_1_0 = "/* XXX 0N0 XXX */\n" + AUTOGENERATED_CODE[0][1][0]
    AUTOGENERATED_CODE_0_1_1 = "/* XXX 0NK XXX */\n" + AUTOGENERATED_CODE[0][1][1]
    AUTOGENERATED_CODE_1_0_0 = "/* XXX M00 XXX */\n" + AUTOGENERATED_CODE[1][0][0]
    AUTOGENERATED_CODE_1_0_1 = "/* XXX M0K XXX */\n" + AUTOGENERATED_CODE[1][0][1]
    AUTOGENERATED_CODE_1_1_0 = "/* XXX MN0 XXX */\n" + AUTOGENERATED_CODE[1][1][0]
    AUTOGENERATED_CODE_1_1_1 = "/* XXX MNK XXX */\n" + AUTOGENERATED_CODE[1][1][1]

    num_tiles_k = DKI_padded
    if K % tile_K != 0:
        num_tiles_k -= 1
        k_remainder = True
    else:
        k_remainder = False
    num_tiles_m = DMI_padded
    # if M % tile_M != 0:
    #     num_tiles_m -= 1
    #     m_remainder = True
    # else:
    m_remainder = False

    # print('K remainder:', k_remainder, 'M remainder:', (M % tile_M) != 0)

    B_desc = dace.float32[K, N]
    if transpose_b:
        B_desc.set_strides_from_layout(0, 1)

    block_n = dace.symbol("block_n")

    @dace.program(auto_optimize=True)
    def bla(A_val: A_val_desc, A_idx: A_idx_desc, B: B_desc, C: C_desc):
        Clocal = np.ndarray([DMI_padded, m, num_vectors_b, v], dtype=np.float32)
        Clocal[:] = 0

        Blocal = np.ndarray([K, num_vectors_b * v], dtype=np.float32)
        if transpose_b:
            # for i, j in dace.map[0:K, 0:num_vectors_b*v]:
            #     Blocal[i, j] = B[i, block_n*num_vectors_b*v + j]
            with dace.tasklet:
                (
                    inB
                    << B[
                        :,
                        block_n * num_vectors_b * v : (block_n + 1) * num_vectors_b * v,
                    ]
                )
                outB >> Blocal[:, :]
                f"""
                copy_with_transpose(outB, inB, {B_desc.strides[1]}, {num_vectors_b*v}, {num_vectors_b*v}, {K});
                """
        else:
            Blocal[:, :] = B[
                :, block_n * num_vectors_b * v : (block_n + 1) * num_vectors_b * v
            ]

        for block_k in range(num_tiles_k):
            for block_m in dace.map[0:num_tiles_m]:
                with dace.tasklet(dace.Language.CPP):
                    ptr_a_val << A_val[block_k, block_m, 0:DK2, 0:g, 0:n]
                    ptr_a_idx << A_idx[block_k, block_m, 0:DK2, 0:g, 0:1]
                    ptr_b << Blocal[block_k * DK2 * g : (block_k + 1) * DK2 * g, :]
                    ptr_cin << Clocal[block_m, :, :, :]
                    AUTOGENERATED_CODE_0_0_0
                    ptr_c >> Clocal[block_m, :, :, :]

            # m remainder
            if m_remainder:
                with dace.tasklet(dace.Language.CPP):
                    ptr_a_val << A_val[block_k, -1, 0:DK2, 0:g, 0:n]
                    ptr_a_idx << A_idx[block_k, -1, 0:DK2, 0:g, 0:1]
                    ptr_b << Blocal[block_k * DK2 * g : (block_k + 1) * DK2 * g, :]
                    ptr_cin << Clocal[-1, :, :, :]
                    AUTOGENERATED_CODE_1_0_0
                    ptr_c >> Clocal[-1, :, :, :]

        # k remainder
        if k_remainder:
            for block_m in dace.map[0:num_tiles_m]:
                with dace.tasklet(dace.Language.CPP):
                    ptr_a_val << A_val[-1, block_m, 0:DK2, 0:g, 0:n]
                    ptr_a_idx << A_idx[-1, block_m, 0:DK2, 0:g, 0:1]
                    ptr_b << Blocal[num_tiles_k * DK2 * g :, :]
                    ptr_cin << Clocal[block_m, :, :, :]
                    AUTOGENERATED_CODE_0_0_1
                    ptr_c >> Clocal[block_m, :, :, :]

            # k/m remainder
            if m_remainder:
                with dace.tasklet(dace.Language.CPP):
                    ptr_a_val << A_val[-1, -1, 0:DK2, 0:g, 0:n]
                    ptr_a_idx << A_idx[-1, -1, 0:DK2, 0:g, 0:1]
                    ptr_b << Blocal[num_tiles_k * DK2 * g :, :]
                    ptr_cin << Clocal[-1, :, :, :]
                    AUTOGENERATED_CODE_1_0_1
                    ptr_c >> Clocal[-1, :, :, :]

        # if transpose_c:
        #     # for i, j, k, l in dace.map[0:DMI_padded, 0:m, 0:num_vectors_b, 0:v]:
        #     #     C[i, j, block_n, k, l] = Clocal[i, j, k, l]
        # else:
        C[:, :, block_n, :, :] = Clocal[:, :, :, :]

    @dace.program(auto_optimize=True)
    def gnm_mult(
        A_val: A_val_desc,
        A_idx: A_idx_desc,
        B: B_desc,
        C: C_desc,
    ):
        for block_n in dace.map[0:Nblocks]:
            bla(A_val, A_idx, B, C, block_n=block_n)

    sdfg = gnm_mult.to_sdfg()
    sdfg.append_global_code(
        """
    #include <immintrin.h>
    #define FOR(i, n) for (int i = 0; i < n; ++i)

inline void copy_with_transpose(
    float* dst, 
    const float* src,
    size_t s,
    size_t d,
    size_t m,
    size_t n
) {
    /*
        src stride [s, 1] shape [m, n]
        dst stride [1, d] shape [n, m]
    */
    size_t i = 0;
    for (; i < m / 4 * 4; i += 4) {
        size_t j = 0;
        for (; j < n / 4 * 4; j += 4) {
            __m128 x0 = _mm_loadu_ps(&src[(i + 0) * s + j]);
            __m128 x1 = _mm_loadu_ps(&src[(i + 1) * s + j]);
            __m128 x2 = _mm_loadu_ps(&src[(i + 2) * s + j]);
            __m128 x3 = _mm_loadu_ps(&src[(i + 3) * s + j]);
            __m128 y0 = _mm_unpacklo_ps(x0, x1);
            __m128 y1 = _mm_unpackhi_ps(x0, x1);
            __m128 y2 = _mm_unpacklo_ps(x2, x3);
            __m128 y3 = _mm_unpackhi_ps(x2, x3);
            __m128 z0 = _mm_movelh_ps(y0, y2);
            __m128 z1 = _mm_movehl_ps(y2, y0);
            __m128 z2 = _mm_movelh_ps(y1, y3);
            __m128 z3 = _mm_movehl_ps(y3, y1);
            _mm_store_ps(&dst[i + (j + 0) * d], z0);
            _mm_store_ps(&dst[i + (j + 1) * d], z1);
            _mm_store_ps(&dst[i + (j + 2) * d], z2);
            _mm_store_ps(&dst[i + (j + 3) * d], z3);
        }
        for (; j < n; j += 4) {    
            __m128i mask_load = (n - j == 3) ? _mm_setr_epi32(-1, -1, -1, 0) : 
                                (n - j == 2) ? _mm_setr_epi32(-1, -1,  0, 0) :
                                /*n - j == 1*/ _mm_setr_epi32(-1,  0,  0, 0);
            __m128 x0 = _mm_maskload_ps(&src[(i + 0) * s + j], mask_load);
            __m128 x1 = _mm_maskload_ps(&src[(i + 1) * s + j], mask_load);
            __m128 x2 = _mm_maskload_ps(&src[(i + 2) * s + j], mask_load);
            __m128 x3 = _mm_maskload_ps(&src[(i + 3) * s + j], mask_load);
            __m128 y0 = _mm_unpacklo_ps(x0, x1);
            __m128 y1 = _mm_unpackhi_ps(x0, x1);
            __m128 y2 = _mm_unpacklo_ps(x2, x3);
            __m128 y3 = _mm_unpackhi_ps(x2, x3);
            __m128 z0 = _mm_movelh_ps(y0, y2);
            __m128 z1 = _mm_movehl_ps(y2, y0);
            __m128 z2 = _mm_movelh_ps(y1, y3);
            __m128 z3 = _mm_movehl_ps(y3, y1);
            _mm_store_ps(&dst[i + (j + 0) * d], z0);
            if (n - j == 1) continue;
            _mm_store_ps(&dst[i + (j + 1) * d], z1);
            if (n - j == 2) continue;
            _mm_store_ps(&dst[i + (j + 2) * d], z2);
        }
    }
    for (; i < m; i += 4) {
        __m128i mask_store = (m - i == 3) ? _mm_setr_epi32(-1, -1, -1, 0) : 
                             (m - i == 2) ? _mm_setr_epi32(-1, -1,  0, 0) :
                             /*m - i == 1*/ _mm_setr_epi32(-1,  0,  0, 0);
        size_t j = 0;
        for (; j < n / 4 * 4; j += 4) {
            __m128 x0 = _mm_loadu_ps(&src[(i + 0) * s + j]);
            __m128 x1 = (m - i > 1) ? _mm_loadu_ps(&src[(i + 1) * s + j]) : _mm_setzero_ps();
            __m128 x2 = (m - i > 2) ? _mm_loadu_ps(&src[(i + 2) * s + j]) : _mm_setzero_ps();
            __m128 x3 = (m - i > 3) ? _mm_loadu_ps(&src[(i + 3) * s + j]) : _mm_setzero_ps();
            __m128 y0 = _mm_unpacklo_ps(x0, x1);
            __m128 y1 = _mm_unpackhi_ps(x0, x1);
            __m128 y2 = _mm_unpacklo_ps(x2, x3);
            __m128 y3 = _mm_unpackhi_ps(x2, x3);
            __m128 z0 = _mm_movelh_ps(y0, y2);
            __m128 z1 = _mm_movehl_ps(y2, y0);
            __m128 z2 = _mm_movelh_ps(y1, y3);
            __m128 z3 = _mm_movehl_ps(y3, y1);
            _mm_maskstore_ps(&dst[i + (j + 0) * d], mask_store, z0);
            _mm_maskstore_ps(&dst[i + (j + 1) * d], mask_store, z1);
            _mm_maskstore_ps(&dst[i + (j + 2) * d], mask_store, z2);
            _mm_maskstore_ps(&dst[i + (j + 3) * d], mask_store, z3);
        }
        for (; j < n; j += 4) {    
            __m128i mask_load = (n - j == 3) ? _mm_setr_epi32(-1, -1, -1, 0) : 
                                (n - j == 2) ? _mm_setr_epi32(-1, -1,  0, 0) :
                                /*n - j == 1*/ _mm_setr_epi32(-1,  0,  0, 0);
            __m128 x0 = _mm_maskload_ps(&src[(i + 0) * s + j], mask_load);
            __m128 x1 = (m - i > 1) ? _mm_maskload_ps(&src[(i + 1) * s + j], mask_load) : _mm_setzero_ps();
            __m128 x2 = (m - i > 2) ? _mm_maskload_ps(&src[(i + 2) * s + j], mask_load) : _mm_setzero_ps();
            __m128 x3 = (m - i > 3) ? _mm_maskload_ps(&src[(i + 3) * s + j], mask_load) : _mm_setzero_ps();
            __m128 y0 = _mm_unpacklo_ps(x0, x1);
            __m128 y1 = _mm_unpackhi_ps(x0, x1);
            __m128 y2 = _mm_unpacklo_ps(x2, x3);
            __m128 y3 = _mm_unpackhi_ps(x2, x3);
            __m128 z0 = _mm_movelh_ps(y0, y2);
            __m128 z1 = _mm_movehl_ps(y2, y0);
            __m128 z2 = _mm_movelh_ps(y1, y3);
            __m128 z3 = _mm_movehl_ps(y3, y1);
            _mm_maskstore_ps(&dst[i + (j + 0) * d], mask_store, z0);
            if (n - j == 1) continue;
            _mm_maskstore_ps(&dst[i + (j + 1) * d], mask_store, z1);
            if (n - j == 2) continue;
            _mm_maskstore_ps(&dst[i + (j + 2) * d], mask_store, z2);
        }
    }
}


    """
    )
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False

    if name is not None:
        sdfg.name = name

    from dace.transformation.auto import auto_optimize

    # sdfg = auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if len(node.map.params) > 1:
                node.map.collapse = 2

    def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
        return next(
            n
            for n, _ in sdfg.all_nodes_recursive()
            if isinstance(n, dace.nodes.MapEntry) and pname in n.params
        )

    def find_all_maps_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
        yield from (
            (n, p)
            for n, p in sdfg.all_nodes_recursive()
            if isinstance(n, dace.nodes.MapEntry) and pname in n.params
        )

    from dace.transformation import helpers as xfutil
    from dace.transformation.dataflow import MapExpansion, InLocalStorage

    map = find_map_by_param(sdfg, "block_m")
    # map_m, map_n = MapExpansion.apply_to(sdfg, map_entry=map)
    if tile > 0:
        xfutil.tile(sdfg, map, (N % tile == 0), False, block_n=tile)
    if tile_2 > 0:
        xfutil.tile(sdfg, map, (N % tile_2 == 0), False, block_n=tile_2)
    # xfutil.permute_map(map, [1, 0])
    # tile_map_m = find_map_by_param(sdfg, 'tile_block_n')

    # Hardcoded above
    # if local_b:
    #     state: dace.SDFGState
    #     for map, state in find_all_maps_by_param(sdfg, 'block_n'):
    #         sd = state.parent
    #         # For each n map, add a new transient and copy to it
    #         name = sd.add_datadesc('local_B', copy.deepcopy(local_B_desc), find_new_name=True)
    #         sd.arrays[name].transient = True
    #         new_node = state.add_access(name)
    #         # Redirect B edge through it
    #         edge = next(e for e in state.out_edges(map) if e.data.data == 'B')
    #         state.remove_edge(edge)
    #         state.add_edge(edge.src, edge.src_conn, new_node, None, edge.data)
    #         state.add_edge(new_node, None, edge.dst, edge.dst_conn, dace.Memlet(data=name))

    # Hardcoded above
    # if local_c:
    #     state: dace.SDFGState
    #     for map, state in find_all_maps_by_param(sdfg, 'block_n'):
    #         exitnode = state.exit_node(map)
    #         sd = state.parent
    #         # For each n map, add a new transient and copy to it
    #         name = sd.add_datadesc('local_C', copy.deepcopy(local_C_desc), find_new_name=True)
    #         sd.arrays[name].transient = True
    #         # Redirect input through it
    #         edge = next(e for e in state.out_edges(map) if e.data.data == 'C')
    #         new_input_node = state.add_access(name)
    #         state.remove_edge(edge)
    #         state.add_edge(edge.src, edge.src_conn, new_input_node, None, edge.data)
    #         state.add_edge(new_input_node, None, edge.dst, edge.dst_conn, dace.Memlet(data=name))
    #         # Redirect output edge through it
    #         edge = next(e for e in state.in_edges(exitnode) if e.data.data == 'C')
    #         new_output_node = state.add_access(name)
    #         state.remove_edge(edge)
    #         state.add_edge(edge.src, edge.src_conn, new_output_node, None, dace.Memlet(data=name))
    #         state.add_edge(new_output_node, None, edge.dst, edge.dst_conn, edge.data)

    # InLocalStorage.apply_to(sdfg, dict(array='A_val'), node_a=tile_map_m, node_b=map)
    # InLocalStorage.apply_to(sdfg, dict(array='A_idx'), node_a=tile_map_m, node_b=map)
    dace.Config.set("compiler", "max_stack_array_size", value=999999999999)

    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            if len(node.map.params) > 1:
                node.map.collapse = 2

    return sdfg.compile()


def test_dace(kernel=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("-m", type=int, default=6)
    parser.add_argument("-g", type=int, default=12)
    parser.add_argument("--trans_b", action="store_true", default=False)
    parser.add_argument("--trans_c", action="store_true", default=False)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile_2", type=int, default=0)
    parser.add_argument(
        "--kernel", type=str, default="best", choices=["best", "avx2", "avx512"]
    )

    parser.add_argument("--local_b", action="store_true", default=False)
    parser.add_argument("--local_c", action="store_true", default=False)
    # TODO: loop order

    args = parser.parse_args()

    kernel: str = kernel or args.kernel
    if kernel == "best":
        try:
            import cpuinfo

            flags = cpuinfo.get_cpu_info()["flags"]
            if "avx512f" in flags:
                kernel = "avx512"
            elif "avx2" in flags:
                kernel = "avx2"
            else:
                raise RuntimeError(
                    "This CPU does not support AVX2, which is the minimum requirement for the kernel to run"
                )
        except (ImportError, ModuleNotFoundError):
            print("To select best kernel automatically, use `pip install py-cpuinfo`.")
            kernel = "avx2"

    # torch.manual_seed(1)
    # shape = (4-1, 6-1)
    # shape_b = (6-1, 32-1)
    M, K, N = 768, 3072, 4096
    # M, K, N = 768, 768, 4096
    # M, K, N = 768, 20*12*10, 4096
    shape = (M, K)
    shape_b = (K, N)
    m, n, g = args.m, args.n, args.g
    # m, n, g = 6, 3, 12
    # m, n, g = 6, 3, 7
    # m, n, g = 10, 1, 24
    # m, n, g = 4, 2, 1
    # m, n, g = 2, 1, 1
    # dense_ten = torch.randint(5, 100, shape, device="cpu", dtype=torch.float32)

    # Hardcoded for now
    args.local_b = True
    args.local_c = True

    print(f"Dimensions: {shape[0]} x {shape[1]} x {shape_b[1]}")
    print(f"B transposed: {args.trans_b}, C transposed: {args.trans_c}")
    print(f"n:m:g microkernel/dace hybrid with {n}:{m}:{g} ({kernel.upper()})")

    # Optimizations
    # Transpose always means storing local tiles
    local_b = args.local_b if not args.trans_b else True
    local_c = args.local_c if not args.trans_c else True
    print("Optimizations:")
    if args.tile > 1:
        print("Tile size:", args.tile)
    if args.tile_2 > 1:
        print("Tile 2 size:", args.tile_2)
    print(f"Local storage: (B: {local_b}, C: {local_c})")

    dense_ten = torch.rand(shape, device="cpu", dtype=torch.float32)
    sparse_dim = 0
    group_dim = 1
    t1 = time.time()
    nm_ten = GroupedNMTensor.from_dense(
        dense_ten,
        n=n,
        m=m,
        sparse_dim=sparse_dim,
        group_size=g,
        group_dim=group_dim,
    )
    t2 = time.time()
    print(f"dense->sparse total time {t2-t1:.2f}")

    densified = nm_ten.to_dense()

    assert is_correct_nm(dense_ten, nm_ten.to_dense(), sparse_dim, n, m)

    # Dims: parts of matrix x number of groups x G x N
    # print("val shape:", nm_ten.val.shape)
    # print("idx shape:", nm_ten.idx.shape)
    # print("original", functools.reduce(lambda x, y: x * y, dense_ten.shape, 1))
    # print("sparse", functools.reduce(lambda x, y: x * y, nm_ten.val.shape, 1))
    print(
        "sparsity",
        functools.reduce(lambda x, y: x * y, nm_ten.val.shape, 1)
        / functools.reduce(lambda x, y: x * y, dense_ten.shape, 1),
    )

    # other = torch.randint(200, 300, shape_b, device="cpu", dtype=torch.float32)
    # other = torch.zeros_like(other)
    # other[1,0] = 1
    other = torch.rand(shape_b, device="cpu", dtype=torch.float32)

    # groups = np.array(nm_ten.nm_strides["order"], dtype=np.int32)

    # print('de', densified)
    # print('ot', other)

    expected = densified @ other
    # print('ex', expected)

    print("Compiling kernel...")

    v = 8 if kernel == "avx2" else 16
    # sdfg = generate_configuration(n, m, g, M, N, K, args.trans_b, args.trans_c, vector_size=v,
    #                               loops=[Loop('n'), Loop('k', sequential=True), Loop('m', sequential=True)])
    # compiled = sdfg.compile()
    compiled = nmg_mult(
        (M, K, N),
        m,
        n,
        g,
        args.trans_b,
        args.trans_c,
        args.tile,
        args.tile_2,
        local_b,
        local_c,
        None,
        kernel,
    )

    print("Compilation complete.")

    # tensor.val is of shape (number of groups=DMI*DKI, chunk_size, group_size, n)
    # tensor.idx is of shape (loop_outer_size, chunk_size, group_size, 1)
    A_val = nm_ten.val
    A_idx = nm_ten.idx

    # Problem dimensions
    perms = math.factorial(m) // math.factorial(n) // math.factorial(m - n)
    DKI_padded = int_ceil(K, perms * g)
    DK_padded = DKI_padded * perms * g
    DMI_padded = math.ceil(M / m)
    DM_padded = DMI_padded * m
    num_vectors_b = 4
    v = 8 if kernel == "avx2" else 16
    Nblocks = N // (num_vectors_b * v)

    if args.trans_b:
        # other_padded = torch.zeros([DK_padded, N], dtype=other.dtype)
        # other_padded[:K, :] = other
        # other_reshaped = other_padded.reshape([DKI_padded, perms, g, Nblocks, num_vectors_b, v])
        # other_permuted = other_reshaped.permute(3, 4, 5, 0, 1, 2)
        B = other.permute(1, 0)
    else:
        # other_padded = torch.zeros([DK_padded, N], dtype=other.dtype)
        # other_padded[:K, :] = other
        # B = other_padded
        B = other

    output = torch.empty([DM_padded, N], dtype=other.dtype)
    if args.trans_c:
        output = output.permute(1, 0).contiguous()

    compiled(
        A_val=A_val.contiguous(),
        A_idx=A_idx.contiguous(),
        B=B.contiguous(),
        C=output.contiguous(),
    )

    if args.trans_c:  # Transpose back for correctness testing
        output = output.permute(1, 0).contiguous()
    result = output[:M, :N]

    avgdiff = (result - expected).abs().sum() / result.numel()
    maxdiff = (result - expected).abs().max()
    median_diff = (result - expected).abs().median()
    diffcount = np.sum(np.where((result - expected).abs() > 1e-2, 1, 0))
    print(
        f"avgdiff {avgdiff:.3f} maxdiff {maxdiff:.3f} median_diff {median_diff:.3f}. Count: {diffcount} / {result.numel()}"
    )

    assert torch.allclose(result, expected)


if __name__ == "__main__":
    test_dace()
    # test_dace('avx2')
    # test_dace('avx512')
    # test_dense_nm_conversion()
    # print("ok")
