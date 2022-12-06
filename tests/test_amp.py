import torch
import sten


@sten.register_sparsifier_implementation(
    sparsifier=sten.ScalarFractionSparsifier,
    inp=torch.Tensor,
    out=sten.MaskedSparseTensor,
)
def sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    sparse_ten = sten.random_mask_sparsify(
        tensor,
        sparsifier.fraction
    )
    return sten.SparseTensorWrapper.wrapped_from_dense(
        sten.MaskedSparseTensor(sparse_ten, inplace_sparsifier=sparsifier),
        tensor,
        grad_fmt,
    )
        
        
@sten.register_sparsifier_implementation(
    sparsifier=sten.SameFormatSparsifier,
    inp=torch.Tensor,
    out=sten.MaskedSparseTensor,
)
def sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    ref_sp = sparsifier.ref_sp_ten.wrapped_tensor.sparsifier
    sparse_ten = sten.random_mask_sparsify(
        tensor,
        ref_sp.fraction
    )
    return sten.SparseTensorWrapper.wrapped_from_dense(
        sten.MaskedSparseTensor(sparse_ten, ref_sp),
        tensor,
        grad_fmt,
    )


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.conv2d,
    inp=[
        torch.Tensor,
        sten.MaskedSparseTensor,
        torch.Tensor
    ],
    out=[(sten.KeepAll, torch.Tensor)]
)
def conv2d_impl(ctx, inputs, output_sparsifiers):
    # autocast_enabled = torch.is_autocast_enabled()
    # with torch.autocast("cuda", enabled=False):
    
    input, weight, bias = inputs
    dense_weight = weight.wrapped_tensor.to_dense()
    
    #if autocast_enabled:
    # perform explicit cast and then call appropriate implementation
    dt = torch.get_autocast_gpu_dtype()
    
    # input = input.to(dtype=dt).clone().detach()
    # dense_weight = dense_weight.to(dtype=dt).clone().detach()
    # bias = bias.to(dtype=dt).clone().detach()

    # output = torch.nn.functional.conv2d(input, dense_weight, bias).to(dtype=torch.float32).clone().detach()
    
    #output = torch.from_numpy(output.cpu().numpy()).cuda()
    
    # input = input.clone().detach()
    # dense_weight = dense_weight.clone().detach()
    # bias = bias.clone().detach()

    # output = torch.nn.functional.conv2d(input, dense_weight, bias).clone().detach()
    
    def fullclone(inp):
        #return torch.from_numpy(inp.cpu().numpy()).cuda()
        #return inp.detach().clone()
        return inp#.clone().detach()
    
    if torch.is_autocast_enabled():
        print('with autocast')
        input = fullclone(input)
        dense_weight = fullclone(dense_weight)
        bias = fullclone(bias)
        output = torch.nn.functional.conv2d(
            input.to(dtype=torch.float16),
            dense_weight.to(dtype=torch.float16),
            bias.to(dtype=torch.float16),
        )#.to(dtype=torch.float32)
        output = fullclone(output)
    else:
        print('without autocast')
        output = torch.nn.functional.conv2d(
            input,
            dense_weight,
            bias,
        )
    
    ctx.save_for_backward(input, dense_weight, bias)
        
    return output


@sten.register_bwd_op_impl(
    operator=torch.nn.functional.conv2d,
    grad_out=[torch.Tensor],
    grad_inp=(
        (sten.KeepAll, torch.Tensor),
        (sten.KeepAll, torch.Tensor),
        (sten.KeepAll, torch.Tensor),
    ),
    inp=(torch.Tensor, sten.MaskedSparseTensor, torch.Tensor),
)
def conv2d_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input, weight, bias = ctx.saved_tensors
        
    # if input.dtype != torch.float16:
    #     raise Exception('asdfjklasrg32iu 34ilut ')
        
    [doutput] = grad_outputs
    
    if doutput.dtype == torch.float16:
        print('bwd with autocast')
        ci = input.detach().clone().to(dtype=torch.float16).requires_grad_(True)
        cw = weight.detach().clone().to(dtype=torch.float16).requires_grad_(True)
        cb = bias.detach().clone().to(dtype=torch.float16).requires_grad_(True)
        with torch.enable_grad():
            output = torch.nn.functional.conv2d(ci, cw, cb)
        output.backward(doutput)
    else:
        print('bwd without autocast')
        ci = input.detach().clone().requires_grad_(True)
        cw = weight.detach().clone().requires_grad_(True)
        cb = bias.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            output = torch.nn.functional.conv2d(ci, cw, cb)
        output.backward(doutput)
    
    dinput = ci.grad
    dweight = cw.grad
    dbias = cb.grad
    
    if doutput.dtype == torch.float16:
        dinput = dinput.to(dtype=torch.float32)
        dweight = dweight.to(dtype=torch.float32)
        dbias = dbias.to(dtype=torch.float32)
        
    return dinput, dweight, dbias


def amp_iteration(model, x):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = model(x)
        l = loss(y, torch.ones_like(y))
    
    #l.backward()
    
    scaler.scale(l).backward()
    if type(model[0].weight) != torch.nn.parameter.Parameter:
        assert type(model[0].weight.grad) != torch.Tensor
    scaler.step(optimizer)
    scaler.update()
    
    return y, x.grad


def test_simple():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3,5,3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(5,7,3),
        torch.nn.ReLU(),
    ).cuda()
    
    sb = sten.SparsityBuilder()
    sb.set_weight(
        name='0.weight',
        initial_sparsifier=sten.ScalarFractionSparsifier(0.0),
        out_format=sten.MaskedSparseTensor,
    )
    sparse_model = sb.get_sparse_model(model)

    x = torch.randn(2, 3, 10, 10, device='cuda', requires_grad=True)
    sp_x = x.detach().clone().requires_grad_(True)

    y, x_grad = amp_iteration(model, x)
    sp_y, sp_x_grad = amp_iteration(sparse_model, sp_x)
    
    assert torch.allclose(y, sp_y)
    assert torch.allclose(x_grad, sp_x_grad)
    assert torch.allclose(model[0].weight, sparse_model[0].weight)
    

if __name__ == '__main__':
    test_simple()