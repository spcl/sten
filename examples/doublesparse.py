import torch
import sten
import transformers
from time import time, sleep
import numpy as np
import math
from double_sparse_compression.inference_kernels.kernel_selector import get_doublesparse_mul
from double_sparse_compression.inference import FeatureFlags


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




class DoubleSparseSparsifier:
    def __init__(self, sparsity, k):
        self.sparsity = sparsity
        self.k = k


class DoubleSparseTensor:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_row_offsets: torch.Tensor,
        a_col_vals: torch.Tensor,
        b_row_offsets: torch.Tensor,
        b_col_vals: torch.Tensor,
        non_zero_rows: int,
        orig_dense: torch.Tensor,
    ):
        self.m = m
        self.n = n
        self.k = k

        self.a_row_offsets = a_row_offsets
        self.a_col_vals = a_col_vals
        self.b_row_offsets = b_row_offsets
        self.b_col_vals = b_col_vals
        self.non_zero_rows = non_zero_rows
        self.workspace = None
        self.orig_dense = orig_dense

    def to_dense(self):
        return self.orig_dense


def generate_x_fp32(n, upper_bound=2):
    return ((torch.rand(n) - 0.5) * upper_bound).round().float()


def create_x_random(n, upper_bound=2):
    return generate_x_fp32(n, upper_bound).half()


def random_csr_host(m, n, density):
    r = ((torch.rand(m, n) <= density) * (create_x_random(m * n).reshape(m, n)).float()).to_sparse_csr()
    return r    

def merge_col_val(col_ids, values):
    """
    Merge 16-bit col ids buffer with the 16-bi values buffer into a single buffer with the columns
    occupying the lower half of the 32-bit number.

    @param col_ids: CSR column ids
    @param values:  CSR values
    @return: Merged colvals buffer.
    """
    if values.shape[0] != 0:
        return (
            values.view(torch.int16)
            .to(torch.int64)
            .bitwise_left_shift(16)
            .bitwise_or(col_ids.view(torch.int16).to(torch.int64))
            .to(torch.int32)
        )
    else:
        return torch.zeros(0)


@sten.register_sparsifier_implementation(
    sparsifier=DoubleSparseSparsifier, inp=torch.Tensor, out=DoubleSparseTensor
)
def double_sparse_sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    device = tensor.device
    
    m, n = tensor.shape
    k = sparsifier.k
    density = 1 - sparsifier.sparsity
    
    a = random_csr_host(m, k, density)
    b_sparse = random_csr_host(k, n, density)
    
    b_row_offsets = b_sparse.crow_indices()
    row_counts = torch.diff(b_row_offsets)
    non_zero_row_count = (row_counts != 0).sum().item()

    row_counts_sorted, row_ids = row_counts.sort(descending=False)

    non_zero_row_ids_sorted = row_ids[row_counts_sorted != 0]
    zero_row_ids_sorted = row_ids[row_counts_sorted == 0]

    a_dense = a.to_dense()

    reordered_row_ids = torch.concat((non_zero_row_ids_sorted, zero_row_ids_sorted))

    a_dense = a_dense[:, reordered_row_ids]
    a_dense[:, non_zero_row_ids_sorted.shape[0]:] = 0
    b_dense = b_sparse.to_dense()[reordered_row_ids, :]
    a_sparse = a_dense.to_sparse_csr()
    b_sparse = b_dense.to_sparse_csr()

    double_sparse_tensor = DoubleSparseTensor(
        m, n, k,
        a_sparse.crow_indices().int(),
        merge_col_val(a_sparse.col_indices().short(), a_sparse.values().half()),
        b_sparse.crow_indices().int(),
        merge_col_val(b_sparse.col_indices().short(), b_sparse.values().half()),
        non_zero_row_count,
        tensor,
    )
    
    return sten.SparseTensorWrapper.wrapped_from_dense(
        double_sparse_tensor,
        tensor,
        grad_fmt,
    )


double_sparse_mul = get_doublesparse_mul()


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, DoubleSparseTensor),
    out=[(sten.KeepAll, torch.Tensor)],
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    input, weight = inputs
    # input shape [*M, K]
    # weight shape [N, K]
    # output shape [*M, N]
    
    double_sparse_tensor = weight.wrapped_tensor
    
    M = math.prod(input.shape[:-1])
    K = input.shape[-1]
    N = double_sparse_tensor.n
    
    x = input.reshape(M, K).T.half().contiguous()
    batch_size = x.shape[1] # M
    y = torch.empty((1, M, N), dtype=torch.float16, device=x.device)
    if double_sparse_tensor.workspace is None:
        double_sparse_tensor.workspace = torch.empty(double_sparse_tensor.k * M, dtype=torch.float32, device=x.device).contiguous()
    workspace = double_sparse_tensor.workspace
    flag = FeatureFlags.CSR_ASYNC
    double_sparse_mul(
        double_sparse_tensor.m,
        double_sparse_tensor.n,
        double_sparse_tensor.k,
        double_sparse_tensor.a_row_offsets,
        double_sparse_tensor.a_col_vals,
        double_sparse_tensor.b_row_offsets,
        double_sparse_tensor.b_col_vals,
        double_sparse_tensor.non_zero_rows,
        batch_size,
        x,
        flag,
        workspace,
        y,
        y
    )
    return y.reshape(input.shape[:-1] + (N,)),

def test_double_sparse():
    dtype = torch.float16
    model_id = "meta-llama/Llama-2-7b-hf"
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
    
    with torch.no_grad():
        mean, std, times = time_prof(3, lambda: model(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime (dense) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")
    
    weights_to_sparsify = []
    sb = sten.SparsityBuilder()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            weight = module_name + ".weight"
            weights_to_sparsify.append(weight)
            sb.set_weight(
                name=weight,
                initial_sparsifier=DoubleSparseSparsifier(sparsity=0.5, k=4096),
                out_format=DoubleSparseTensor,
            )
    print(weights_to_sparsify)
    
    model = sb.sparsify_model_inplace(model)
    
    # this is required to avoid out of memory error when invoking the sparse model
    with torch.no_grad():
        sparse_output = model(random_input, **kwargs)

    with torch.no_grad():
        mean, std, times = time_prof(3, lambda: model(random_input, **kwargs), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime (double_sparse) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")


if __name__ == "__main__":
    test_double_sparse()
