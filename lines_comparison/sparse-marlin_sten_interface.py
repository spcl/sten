# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/tests/test_sparse_marlin.py
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