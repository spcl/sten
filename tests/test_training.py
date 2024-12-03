# Following example https://pytorch.org/blog/accelerating-neural-network-training/
import torch
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)
import sten
from transformers import AutoModel


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
    
    # y = x * w.t -> y = (w * x.t).t
    # grad_x = grad_y * w -> grad_x = (w.t * grad_y.t).t
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
    
    # grad_x = w.t @ grad_y2d
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


def run_parametrized(enable_compile, backend):
    assert torch.cuda.is_available()

    device = torch.device("cuda")

    model = AutoModel.from_pretrained('facebook/dinov2-large').to(device).half()
    model.train()

    batch = 32
    channels = 3
    height = 224
    width = 224
    repeats = 10
    
    def train_iteration(model, optimizer):
        inputs = torch.randn((batch, channels, height, width), dtype=torch.half, device=device)
        outputs = model(inputs)
        loss = outputs.last_hidden_state.mean()
        loss = loss * loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    
    assert backend in ['dense', 'torchao', 'sten']
    if backend == 'dense':
        pass
    elif backend == 'torchao':
        sparse_config = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                sparse_config[name] = SemiSparseLinear
        swap_linear_with_semi_sparse_linear(model, sparse_config)
    elif backend == 'sten':
        sb = sten.SparsityBuilder()
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module_name + ".weight"
                sb.set_weight(
                    name=weight,
                    initial_sparsifier=TrainingSparsifier(),
                    out_format=TrainingTensor,
                )
        sb.sparsify_model_inplace(model)
        
    if enable_compile:
        model = torch.compile(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    mean, std, times = sten.time_prof(repeats, lambda: train_iteration(model, optimizer), sync=torch.cuda.synchronize, warmup=0.3)
    print(f"Runtime ({backend}, compile={enable_compile}) [ms] mean: {mean * 1e3:.3f}, std: {std * 1e3:.3f}, repeats: {len(times)}")        


def test_training():
    run_parametrized(False, 'dense')
    run_parametrized(False, 'torchao')
    run_parametrized(False, 'sten')
    run_parametrized(True, 'dense')
    run_parametrized(True, 'torchao')
    run_parametrized(True, 'sten')


if __name__ == "__main__":
    test_training()