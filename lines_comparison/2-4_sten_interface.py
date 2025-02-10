# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/tests/test_training.py
class TrainingSparsifier:
    def __init__(self):
        pass
    

class TrainingTensor:
    def __init__(self, weight):
        self.weight = weight
        
    def to_dense(self):
        return self.weight
    
    def add_(self, other, alpha=1):
        self.weight.add_(other, alpha=alpha)
        
    @property
    def shape(self):
        return self.weight.shape


@sten.register_sparsifier_implementation(
    sparsifier=TrainingSparsifier, inp=torch.Tensor, out=TrainingTensor
)
def sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    return sten.SparseTensorWrapper.wrapped_from_dense(
        TrainingTensor(tensor),
        tensor,
        grad_fmt,
    )


@torch.compiler.allow_in_graph
def sparse_semi_structured_tile_wrapper(tensor):
    return torch._sparse_semi_structured_tile(tensor, algorithm='', use_cutlass=False)


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, TrainingTensor, torch.Tensor),  
    out=None,  # default (dense)
)
def operator_impl(ctx, inputs, output_sparsifiers):
    x, w, bias = inputs
    dense_weight = w.wrapped_tensor.weight
    
    (packed, meta, packed_t, meta_t, bitmask) = sparse_semi_structured_tile_wrapper(dense_weight)

    x_shape = x.shape
    x2d = x.view(-1, x.shape[-1])

    row, col = x2d.shape
    x_padded = torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT._pad_dense_input(x2d)
    
    result_padded = torch._cslt_sparse_mm(
        packed,
        x_padded.t(),
        bias=bias,
        transpose_result=True,#False,
        alg_id=torch.sparse.semi_structured.SparseSemiStructuredTensor._DEFAULT_ALG_ID,
    )#.t()
    result = result_padded[:row, :]

    ctx.save_for_backward(x, packed_t)

    return result.view(*x_shape[:-1], -1)


@sten.register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=None,  # default (dense)
    grad_inp=None,  # default (dense)
    inp=(torch.Tensor, TrainingTensor, torch.Tensor),  
)
def my_operator(ctx, grad_outputs, input_sparsifiers):
    x, packed_t = ctx.saved_tensors
    [grad_y] = grad_outputs
    grad_y_shape = grad_y.shape
    grad_y2d = grad_y.reshape(-1, grad_y.shape[-1])
    x2d = x.view(-1, x.shape[-1])
    grad_w = torch.mm(grad_y2d.t(), x2d)
    
    row, col = grad_y2d.shape
    grad_y2d_padded = torch.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT._pad_dense_input(grad_y2d)
    
    grad_x2d_padded = torch._cslt_sparse_mm(
        packed_t,
        grad_y2d_padded.t(),
        bias=None,
        transpose_result=True,#False,
        alg_id=torch.sparse.semi_structured.SparseSemiStructuredTensor._DEFAULT_ALG_ID,
    )#.t()
    
    grad_x2d = grad_x2d_padded[:row, :]
    grad_x = grad_x2d.view(*grad_y_shape[:-1], -1)
        
    grad_b = grad_y.sum(dim=list(range(grad_y.ndim - 1)))
    grad_inputs = (grad_x, grad_w, grad_b)
    return grad_inputs