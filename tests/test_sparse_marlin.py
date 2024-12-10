import torch
import sten
import marlin
import transformers
from time import time, sleep
import numpy as np

def time_prof(repeats, func, init=lambda: (), sync=lambda: (), number=1, warmup=0.3, time_budget=0.1, cooldown=0.1):    
    times = []
    
    min_warmup_repeats = int(np.ceil(repeats * warmup))
    min_total_repeats = min_warmup_repeats + repeats
        
    i = 0
    total_elapsed = 0
    while True:
        init()
        t1 = time()
        for _ in range(number):
            func()
        sync()
        t2 = time()
        # cooldown
        sleep(cooldown)
        elapsed = (t2 - t1) / number
        total_elapsed += elapsed
        times.append(elapsed)
        i += 1
        if total_elapsed > time_budget and i >= min_total_repeats:
            break
    
    assert len(times) >= min_total_repeats
    times = times[int(np.ceil((len(times) - min_warmup_repeats) * warmup)):]
    assert len(times) >= repeats
    
    mean = np.mean(times)
    std = np.std(times)
    
    return mean, std, times


def gen_2_4(n, k, dev):
    B = torch.randint(low=-(2**31), high=2**31, size=(k * n // 8 // 2,), device=dev)
    meta = torch.ones((n * k // 16,), dtype=torch.int16, device=dev) * (-4370)
    return B, meta


def get_problem_24(dev, n, k, groupsize=-1):
    B, meta = gen_2_4(n, k, dev)
    if groupsize == -1:
        s = torch.zeros((1, n), dtype=torch.half, device=dev)
    else:
        s = torch.zeros(
            (((k // 2) // (groupsize // 2)), n), dtype=torch.half, device=dev
        )
    torch.cuda.synchronize()
    return B, s, meta


class SparseMarlinSparsifier:
    def __init__(self):
        pass


class SparseMarlinTensor:
    def __init__(self, B, s, meta, workspace, orig_dense):
        self.B = B
        self.s = s
        self.meta = meta
        self.workspace = workspace
        self.orig_dense = orig_dense

    def to_dense(self):
        return self.orig_dense


@sten.register_sparsifier_implementation(
    sparsifier=SparseMarlinSparsifier, inp=torch.Tensor, out=SparseMarlinTensor
)
def marlin_sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    device = tensor.device
    # groupsize = 128
    groupsize = -1
    max_par = 16
    n, k = tensor.shape
    B, s, meta = get_problem_24(device, n, k, groupsize)
    workspace = torch.zeros(n // 128 * max_par, device=device, dtype=torch.int32)
    return sten.SparseTensorWrapper.wrapped_from_dense(
        SparseMarlinTensor(B, s, meta, workspace, tensor),
        tensor,
        grad_fmt,
    )


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, SparseMarlinTensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    input1, input2 = inputs
    A = input1
    *m, k = A.shape
    device = input2.wrapped_tensor.orig_dense.device
    n, k1 = input2.wrapped_tensor.orig_dense.shape
    assert k == k1
    B = input2.wrapped_tensor.B
    s = input2.wrapped_tensor.s
    meta = input2.wrapped_tensor.meta
    workspace = input2.wrapped_tensor.workspace
    C = torch.empty((*m, n), dtype=torch.half, device=device)
    thread_k, thread_m, sms, max_par = -1, -1, -1, 16
    marlin.mul_2_4(A.view(-1, k), B, meta, C.view(-1, n), s, workspace, thread_k, thread_m, sms, max_par)
    return C


def test_sparse_marlin():
    dtype = torch.float16
    model_id = "meta-llama/Llama-3.1-8B"
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": dtype}, device_map="auto"
    )
    batch = 8
    seq_len = 512
    feature_dim = 4096
    head_dim = 128
    # random_input = torch.randint(0, 100, (batch, seq_len))
    
    random_input = torch.randn((batch, seq_len, feature_dim), dtype=dtype, device="cuda:0")
    position_embeddings = (
        torch.randn((batch, seq_len, head_dim), dtype=dtype, device="cuda:0"), 
        torch.randn((batch, seq_len, head_dim), dtype=dtype, device="cuda:0")
    )
    
    kwargs = {"position_embeddings": position_embeddings}
    
    model = pipeline.model.model.layers[0]
    # model = pipeline.model
    # model(random_input, position_embeddings=position_embeddings)  # returns tuple with single element -- output tensor
        
    weights_to_sparsify = []
    sb = sten.SparsityBuilder()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            weight = module_name + ".weight"
            weights_to_sparsify.append(weight)
            sb.set_weight(
                name=weight,
                initial_sparsifier=SparseMarlinSparsifier(),
                out_format=SparseMarlinTensor,
            )
    print(weights_to_sparsify)
    
    print(f'before invoking dense model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    with torch.no_grad():
        dense_output = model(random_input, **kwargs)
    print(f'after invoking dense model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')

    model = sb.sparsify_model_inplace(model)
    print(f'after sparsifying model: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    
    # this is required to avoid out of memory error when invoking the sparse model
    with torch.no_grad():
        sparse_output = model(random_input, **kwargs)
    print(f'after invoking sparse model with no grad: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')

    with torch.no_grad():
        mean, std, times = time_prof(3, lambda: model(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")


if __name__ == "__main__":
    test_sparse_marlin()
