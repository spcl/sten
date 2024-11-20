import torch
import sten
import transformers
import sparse_cuda


class SparsegenSparsifier:
    def __init__(self, n, m):
        self.n = n
        self.m = m


class SparsegenTensor:
    def __init__(self, n, m, blockheight, blockwidth, packed_weight, packed_mask, orig_dense):
        self.n = n
        self.m = m
        self.blockheight = blockheight
        self.blockwidth = blockwidth
        self.packed_weight = packed_weight
        self.packed_mask = packed_mask
        self.orig_dense = orig_dense

    def to_dense(self):
        return self.orig_dense


def make_sparse(w, n, m):
    w = w.t()
    w1 = w.reshape((-1, m))
    idx = torch.topk(torch.abs(w1), m - n, dim=1, largest=False)[1]
    w1 = torch.scatter(w1, 1, idx, 0)
    w1 = w1.reshape(w.shape).t()
    return w1.contiguous()


def pack(w):
    mask = w != 0
    packed = w.t()[mask.t()]
    packed = packed.reshape((w.shape[1], -1)).t()
    packed = packed.contiguous()

    mask = np.asarray(mask.cpu(), dtype=np.uint32)
    packed_mask = np.zeros((mask.shape[0] // 32, mask.shape[1]), dtype=np.uint32)
    for i in range(packed_mask.shape[0]):
        for j in range(32):
            packed_mask[i] |= mask[32 * i + j] << j
    packed_mask = packed_mask.astype(np.int32)
    packed_mask = torch.from_numpy(packed_mask)

    return packed.to(w.device), packed_mask.to(w.device)


@sten.register_sparsifier_implementation(
    sparsifier=SparsegenSparsifier, inp=torch.Tensor, out=SparsegenTensor
)
def marlin_sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    w1 = tensor.t()
    w_sparse = make_sparse(w1, sparsifier.n, sparsifier.m)
    w_packed, packed_mask = pack(w_sparse)
    blockheight = 256
    blockwidth = 64
    return sten.SparseTensorWrapper.wrapped_from_dense(
        SparsegenTensor(
            n=sparsifier.n, m=sparsifier.m, blockheight=blockheight, blockwidth=blockwidth,
            packed_weight=w_packed, packed_mask=packed_mask, orig_dense=tensor,
        ),
        tensor,
        grad_fmt,
    )


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, SparsegenTensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    x, w = inputs
    wt = w.wrapped_tensor
    x = x.to(wt.orig_dense.device).bfloat16()
    y = torch.empty((*x.shape[:-1], wt.orig_dense.shape[0]), dtype=torch.bfloat16, device=x.device)
    x2d = x.view(-1, x.shape[-1])
    y2d = y.view(-1, y.shape[-1])
    for m in range(y2d.shape[0]):
        # print('Trying to run', x2d.shape, wt.packed_weight.shape, wt.packed_mask.shape, y2d.shape)
        # print('Devices:', x2d.device, wt.packed_weight.device, wt.packed_mask.device, y2d.device)
        sparse_cuda.vecsparsematmul(
            x2d[m], wt.packed_weight, wt.packed_mask, y2d[m], wt.n, wt.m, wt.blockheight, wt.blockwidth
        )
        # print('after kernel launch')
        # torch.cuda.synchronize()
        # print('after sync')
    return y


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


def test_sparsegen():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    # model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    model_id = "meta-llama/Llama-3.1-8B"
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": dtype}, device_map="auto"
    )
    batch = 8
    seq_len = 512
    feature_dim = 4096
    head_dim = 128
    # random_input = torch.randint(0, 100, (batch, seq_len))
    
    sparse_n = 16
    sparse_m = 64
    
    random_input = torch.randn((batch, seq_len, feature_dim), dtype=dtype, device=device)
    position_embeddings = (
        torch.randn((batch, seq_len, head_dim), dtype=dtype, device=device), 
        torch.randn((batch, seq_len, head_dim), dtype=dtype, device=device)
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
                initial_sparsifier=SparsegenSparsifier(n=sparse_n, m=sparse_m),
                out_format=SparsegenTensor,
            )
            # break
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
        # sparse_output = model(random_input)
    print(f'after invoking sparse model with no grad: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    
    # assert torch.allclose(dense_output.logits, sparse_output.logits, atol=1e-3)
    # print('outputs are close')

    with torch.no_grad():
        # mean, std, times = time_prof(20, lambda: model(random_input, position_embeddings=position_embeddings), sync=torch.cuda.synchronize, warmup=0.3)
        mean, std, times = time_prof(3, lambda: model(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")


if __name__ == "__main__":
    test_sparsegen()
