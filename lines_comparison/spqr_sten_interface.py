# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/tests/test_spqr.py
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