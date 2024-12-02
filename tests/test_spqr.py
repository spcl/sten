import torch
import sten
import transformers
from dataclasses import dataclass

import spqr
import functools

# https://stackoverflow.com/a/31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class SpQRSparsifier:
    def __init__(self, n, m, bits, beta1, beta2, dense_weights, row_offsets, col_vals, in_perm):
        self.n = n
        self.m = m
        self.bits = bits
        self.beta1 = beta1
        self.beta2 = beta2
        self.dense_weights = dense_weights
        self.row_offsets = row_offsets
        self.col_vals = col_vals
        self.in_perm = in_perm


class SpQRTensor:
    def __init__(self, n, m, bits, beta1, beta2, dense_weights, row_offsets, col_vals, in_perm, orig_dense):
        self.n = n
        self.m = m
        self.bits = bits
        self.beta1 = beta1
        self.beta2 = beta2
        self.dense_weights = dense_weights
        self.row_offsets = row_offsets
        self.col_vals = col_vals
        self.in_perm = in_perm
        self.orig_dense = orig_dense

    def to_dense(self):
        return self.orig_dense


@sten.register_sparsifier_implementation(
    sparsifier=SpQRSparsifier, inp=torch.Tensor, out=SpQRTensor
)
def marlin_sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    return sten.SparseTensorWrapper.wrapped_from_dense(
        SpQRTensor(
            sparsifier.n,
            sparsifier.m,
            sparsifier.bits,
            sparsifier.beta1,
            sparsifier.beta2,
            sparsifier.dense_weights,
            sparsifier.row_offsets,
            sparsifier.col_vals,
            sparsifier.in_perm,
            tensor,
        ),
        tensor,
        grad_fmt,
    )


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, SpQRTensor, torch.Tensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    x, w, b = inputs
    wt = w.wrapped_tensor
    x = x.to(wt.orig_dense.device).bfloat16()
    y = torch.empty((*x.shape[:-1], wt.orig_dense.shape[0]), dtype=torch.bfloat16, device=x.device)
    x2d = x.view(-1, x.shape[-1])
    y2d = y.view(-1, y.shape[-1])
    nnz = wt.col_vals.shape[0]
    should_reorder = wt.in_perm is not None and torch.numel(wt.in_perm) != 0
    for m in range(y2d.shape[0]):
        x2dm = x2d[m]
        if should_reorder:
            x2dm = x2dm[wt.in_perm]
        spqr.call_spqr_mul(
            wt.m,
            wt.n,
            wt.bits,
            wt.beta1,
            wt.beta2,
            wt.dense_weights,
            wt.row_offsets,
            wt.col_vals,
            nnz,
            x2dm,
            int(spqr.FeatureFlags.SPARSE_FUSED_FP32_ASYNC),
            y2d[m],
            y2d[m],
        )
    if b is not None:
        output = y + b
    else:
        output = y
    return output


from time import time, sleep
import numpy as np


def test_spqr():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    model_id = "facebook/opt-125m"
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": dtype}, device_map="auto"
    )
        
    batch = 4
    seq_len = pipeline.model.config.max_position_embeddings
    print("seq_len:", seq_len)
    feature_dim = pipeline.model.config.hidden_size
    print("feature_dim:", feature_dim)
    random_input = torch.randn((batch, seq_len, feature_dim), dtype=dtype, device=device)
    
    kwargs = {}
    
    model = pipeline.model
        
    model_layer = pipeline.model.model.decoder.layers[0]
    random_layer_input = torch.randn((batch, seq_len, feature_dim), dtype=dtype, device=device)

    [layer_output] = model_layer(random_layer_input)
    
    quantized_model_path = '/users/aivanov/SpQR/quantized_opt-125m-inference.pt'
    quantized_model = torch.load(quantized_model_path)
    quantized_model.to(device)
    quantized_model_layer = quantized_model.model.decoder.layers[0]
    # args = SpQRArgs(seqlen=seq_len)
    
    # dataloader = get_loaders(
    #     args.dataset,
    #     nsamples=args.nsamples,
    #     seed=args.seed,
    #     model_path=args.model_path,
    #     seqlen=args.seqlen,
    # )
    # results = quantize_spqr(model, dataloader, args, device)
    
    weights_to_sparsify = []
    sb = sten.SparsityBuilder()
    for module_name, module in model_layer.named_modules():
        if module_name == '':
            continue
        qlm = rgetattr(quantized_model_layer, module_name)
        if isinstance(qlm, spqr.QuantizedLinear):
            weight = module_name + ".weight"
            weights_to_sparsify.append(weight)
            sb.set_weight(
                name=weight,
                initial_sparsifier=SpQRSparsifier(
                    m=qlm.m,
                    n=qlm.n,
                    bits=qlm.bits,
                    beta1=qlm.beta1,
                    beta2=qlm.beta2,
                    dense_weights=qlm.dense_weights,
                    row_offsets=qlm.row_offsets,
                    col_vals=qlm.col_vals,
                    in_perm=qlm.in_perm,
                ),
                out_format=SpQRTensor,
            )
    print(weights_to_sparsify)
    
    print(f'before invoking dense model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    with torch.no_grad():
        mean, std, times = time_prof(3, lambda: model_layer(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime (dense) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")
    print(f'after invoking dense model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')

    model_layer = sb.sparsify_model_inplace(model_layer)
    print(f'after sparsifying model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    
    # this is required to avoid out of memory error when invoking the sparse model
    with torch.no_grad():
        sparse_output = model_layer(random_input, **kwargs)
    print(f'after invoking sparse model with no grad: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')

    with torch.no_grad():
        mean, std, times = time_prof(3, lambda: model_layer(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime (spqr) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")


if __name__ == "__main__":
    test_spqr()
