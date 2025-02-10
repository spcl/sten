# https://github.com/pytorch/ao/blob/4d1c7741842a1dfbd479b3481fcdc93c64db703e/torchao/sparsity/training/__init__.py

class SemiSparseLinear(torch.nn.Linear):
    """
    Replacement nn.Linear that supports runtime weight sparsity
    """

    def forward(self, x):
        sparse_weight = semi_structured_sparsify(self.weight, backend="cusparselt")
        return torch.nn.functional.linear(x, sparse_weight, self.bias)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    @classmethod
    def to_dense(cls, semi_sparse_linear):
        mod = torch.nn.Linear(
            semi_sparse_linear.in_features, semi_sparse_linear.out_features
        )
        mod.weight = semi_sparse_linear.weight
        mod.bias = semi_sparse_linear.bias
        return mod


class SemiSparseActivationLinear(torch.nn.Linear):
    """
    Replacement nn.Linear that supports runtime activation sparsity
    """

    def forward(self, x):
        sparse_x = semi_structured_sparsify(x, backend="cusparselt")
        return torch.nn.functional.linear(sparse_x, self.weight, self.bias)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.in_features, linear.out_features)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod

    @classmethod
    def to_dense(cls, semi_sparse_linear):
        mod = torch.nn.Linear(
            semi_sparse_linear.in_features, semi_sparse_linear.out_features
        )
        mod.weight = semi_sparse_linear.weight
        mod.bias = semi_sparse_linear.bias
        return mod


def swap_linear_with_semi_sparse_linear(model, config, current=""):
    """
    Public API for replacing nn.Linear with SemiSparseLinear
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, torch.nn.Linear):
            if fqn in config:
                setattr(model, name, config[fqn].from_dense(child))
                del child
        else:
            swap_linear_with_semi_sparse_linear(child, config, current=fqn)


def swap_semi_sparse_linear_with_linear(model, current=""):
    """
    Public API for replacing instances of SemiSparseLinear/SemiSparseActivaitonLinear with nn.Linear
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, (SemiSparseLinear, SemiSparseActivationLinear)):
            setattr(model, name, child.to_dense(child))
            del child
        else:
            swap_semi_sparse_linear_with_linear(child, current=fqn)

# https://github.com/pytorch/ao/blob/main/torchao/sparsity/training/autograd.py
class _SparsifyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, backend: GRADIENT_TYPE):  # type: ignore[override]
        use_cutlass = backend == "cutlass"
        if not isinstance(x, SparseSemiStructuredTensor):
            (packed, meta, packed_t, meta_t, bitmask) = (
                torch._sparse_semi_structured_tile(
                    x, algorithm=algo, use_cutlass=use_cutlass
                )
            )
            cls = (
                SparseSemiStructuredTensorCUTLASS
                if use_cutlass
                else SparseSemiStructuredTensorCUSPARSELT
            )
            out = cls(
                x.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                compressed_swizzled_bitmask=bitmask,
                requires_grad=False,
                fuse_transpose_cusparselt=True,
            )
        else:
            out = x.detach()

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out, None, None


class _SparsifyLikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        pattern: SparseSemiStructuredTensor,
        gradient=GRADIENT_TYPE.SPARSE,
    ):  # type: ignore[override]
        assert isinstance(pattern, SparseSemiStructuredTensor)

        if not isinstance(pattern, SparseSemiStructuredTensorCUTLASS):
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.compressed_swizzled_bitmask.is_contiguous():
            raise NotImplementedError(
                "`sparsify_like(x, pattern)` is not implemented when `bitmask` is transposed"
            )

        packed, packed_t = torch._sparse_semi_structured_apply(
            x, pattern.compressed_swizzled_bitmask
        )

        # save for backwards
        ctx.meta = pattern.meta
        ctx.meta_t = pattern.meta_t
        ctx.bitmask = pattern.compressed_swizzled_bitmask
        ctx.gradient = gradient

        return pattern.__class__(
            x.shape,
            packed,
            pattern.meta,
            packed_t,
            pattern.meta_t,
            pattern.compressed_swizzled_bitmask,
            requires_grad=x.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx.gradient == GRADIENT_TYPE.STE or isinstance(
            grad_out, SparseSemiStructuredTensor
        ):
            return grad_out, None, None, None
        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype

        if ctx.gradient == GRADIENT_TYPE.DENSE:
            assert ctx.threads_masks.is_contiguous()
            return (
                torch._sparse_semi_structured_apply_dense(grad_out, ctx.bitmask),
                None,
                None,
                None,
            )
        assert ctx.gradient == GRADIENT_TYPE.SPARSE

        packed, _, packed_t, _ = torch._sparse_semi_structured_tile(
            grad_out, ctx.bitmask, backend="cutlass"
        )
        return (
            SparseSemiStructuredTensorCUTLASS(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.bitmask,
                requires_grad=grad_out.requires_grad,
            ),
            None,
            None,
            None,
        )
        return grad_out, None


@torch._dynamo.allow_in_graph
def semi_structured_sparsify(
    x: torch.Tensor,
    algo: str = "",
    backend: str = "cutlass",
) -> SparseSemiStructuredTensor:
    return _SparsifyFunc.apply(x, algo, backend)


@torch._dynamo.allow_in_graph
def semi_structured_sparsify_like(
    x: torch.Tensor,
    pattern: SparseSemiStructuredTensor,
    gradient: GRADIENT_TYPE = GRADIENT_TYPE.SPARSE,
) -> SparseSemiStructuredTensor:
    return _SparsifyLikeFunc.apply(x, pattern, gradient)