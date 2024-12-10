import torch
import sten
import transformers
from dataclasses import dataclass
import os
import spqr_quant as spqr
import functools
import pathlib
from spqr_quant.inference_kernels.cuda_kernel import call_spqr_mul, call_spqr_mul_fused
from spqr_quant.inference import FeatureFlags

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
    inp=(torch.Tensor, SpQRTensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    x, w = inputs
    wt = w.wrapped_tensor
    x = x.to(wt.orig_dense.device).half()
    y = torch.empty((*x.shape[:-1], wt.orig_dense.shape[0]), dtype=torch.float16, device=x.device)
    x2d = x.view(-1, x.shape[-1])
    y2d = y.view(-1, y.shape[-1])
    nnz = wt.col_vals.shape[0]
    should_reorder = wt.in_perm is not None and torch.numel(wt.in_perm) != 0
    for m in range(y2d.shape[0]):
        if should_reorder:
            call_spqr_mul_fused(
                wt.m,
                wt.n,
                wt.bits,
                wt.beta1,
                wt.beta2,
                wt.in_perm,
                wt.dense_weights,
                wt.row_offsets,
                wt.col_vals,
                nnz,
                x2d[m],
                int(FeatureFlags.SPARSE_FUSED_FP32_ASYNC),
                y2d[m],
                y2d[m],
            )
        else:
            call_spqr_mul(
                wt.m,
                wt.n,
                wt.bits,
                wt.beta1,
                wt.beta2,
                wt.dense_weights,
                wt.row_offsets,
                wt.col_vals,
                nnz,
                x2d[m],
                int(FeatureFlags.SPARSE_FUSED_FP32_ASYNC),
                y2d[m],
                y2d[m],
            )
        
    output = y
    return output


from time import time, sleep
import numpy as np


def test_spqr():
    dtype = torch.float16
    device = torch.device("cuda:0")
    model_id = "meta-llama/Llama-2-7b-hf"
    # pipeline = transformers.pipeline(
    #     "text-generation", model=model_id, model_kwargs={"torch_dtype": dtype}, #device_map="auto"
    #     device=device
    # )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True, device_map=device)

    # batch = 4
    # seq_len = pipeline.model.config.max_position_embeddings
    batch = 1
    seq_len = 1
    # print("seq_len:", seq_len)
    # feature_dim = pipeline.model.config.hidden_size
    # print("feature_dim:", feature_dim)
    # random_input = torch.randn((batch, seq_len, feature_dim), dtype=dtype, device=device)
    # random_input = torch.randn((batch, seq_len, feature_dim), dtype=torch.float16, device="cuda:0")
    # position_embeddings = (
    #     torch.randn((batch, seq_len, head_dim), dtype=dtype, device="cuda:0"), 
    #     torch.randn((batch, seq_len, head_dim), dtype=dtype, device="cuda:0")
    # )
    
    kwargs = {}
    # kwargs = {"position_embeddings": position_embeddings}

    # model = pipeline.model
    model.eval()
    
    # random_layer_input = torch.randn((batch, seq_len, feature_dim), dtype=dtype, device=device)

    # [layer_output] = model(random_layer_input)
    
    quantized_model_path = os.environ.get('QUANTIZED_MODEL_PATH', pathlib.Path(__file__).parent.parent.parent / 'SpQR' / 'quantized_llama2-7b-inference.pt')
    quantized_model = torch.load(quantized_model_path)
    quantized_model.to(device)
    # quantized_model = quantized_model.model.decoder.layers[0]
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
    for module_name, module in model.named_modules():
        if module_name == '':
            continue
        qlm = rgetattr(quantized_model, module_name)
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
    # sb.sparsify_model_inplace(model)
    
    total_start = time()
    
    with torch.no_grad():
        max_new_tokens = 128
        # seq_len = 10
        # input_ids = torch.randint(0, 100, (batch, seq_len), device=device)
        
        text = "The recipe for banana bread is "
        input_ids = tokenizer(text, return_tensors="pt").to(device=device).input_ids
        
        seq_len = input_ids.shape[1]
        
        cache_position = torch.arange(seq_len, dtype=torch.int64, device=device)
        generated_ids = torch.zeros(1, seq_len + max_new_tokens * 2, dtype=torch.int, device=device)
        generated_ids[:, cache_position] = input_ids.to(device).to(torch.int)

        past_key_values = transformers.StaticCache(
            model.config, 1, seq_len + max_new_tokens * 2 + 1, device=device, dtype=torch.float16
        )
        logits = model(
            input_ids, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0]
        next_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)
        generated_ids[:, [seq_len]] = next_token
    
        # Generate tokens one by one
        cache_position = torch.tensor([seq_len + 1], device=device)
        timings_s = []
        for _ in range(1, max_new_tokens):
            start_time = time()
            logits = model(
                next_token.clone(),
                position_ids=None,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )[0]
            next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
            generated_ids[:, cache_position] = next_token.int()
            end_time = time()
            # print(f"duration = {end_time - start_time}")
            timings_s.append(end_time - start_time)
            cache_position += 1
            
            
    total_end = time()
    generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)
    print(f"total duration = {total_end - total_start:.6f}")
    
    durations = np.array(timings_s[16:])
    
    print(f"Mean duration after caching initial input = {durations.mean():.6f}")
    print(f"Median duration after caching initial input = {np.median(durations):.6f}")
    print(f"Best duration after caching initial input = {np.min(durations):.6f}")
    
    
    # print(f'before invoking dense model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    # with torch.no_grad():
    #     mean, std, times = sten.time_prof(3, lambda: model(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    # print(f"Runtime (dense) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")
    # print(f'after invoking dense model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')

    # model = sb.sparsify_model_inplace(model)
    # print(f'after sparsifying model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    
    # # this is required to avoid out of memory error when invoking the sparse model
    # with torch.no_grad():
    #     sparse_output = model(random_input, **kwargs)
    # print(f'after invoking sparse model with no grad: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')

    # with torch.no_grad():
    #     mean, std, times = sten.time_prof(3, lambda: model(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    # print(f"Runtime (spqr) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")


if __name__ == "__main__":
    test_spqr()
