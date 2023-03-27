import torch
import sten


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.conv2d,
    inp=[torch.Tensor, sten.MaskedSparseTensor, torch.Tensor],
    out=[(sten.KeepAll, torch.Tensor)],
)
def conv2d_impl(ctx, inputs, output_sparsifiers):
    input, weight, bias = inputs
    dense_weight = weight.wrapped_tensor.to_dense()

    if torch.is_autocast_enabled():
        # perform explicit cast and then call appropriate implementation
        dt = torch.get_autocast_gpu_dtype()
        output = torch.nn.functional.conv2d(
            input.to(dtype=dt),
            dense_weight.to(dtype=dt),
            bias.to(dtype=dt),
        )
    else:
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

    [doutput] = grad_outputs

    if doutput.dtype == torch.float16:
        ci = input.detach().clone().to(dtype=torch.float16).requires_grad_(True)
        cw = weight.detach().clone().to(dtype=torch.float16).requires_grad_(True)
        cb = bias.detach().clone().to(dtype=torch.float16).requires_grad_(True)
        with torch.enable_grad():
            output = torch.nn.functional.conv2d(ci, cw, cb)
        output.backward(doutput)
    else:
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

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        l = loss(y, torch.ones_like(y))

    scaler.scale(l).backward()
    if type(model[0].weight) != torch.nn.parameter.Parameter:
        assert type(model[0].weight.grad) != torch.Tensor
    scaler.step(optimizer)
    scaler.update()

    return y, x.grad


def test_simple():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 5, 3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(5, 7, 3),
        torch.nn.ReLU(),
    ).cuda()

    sb = sten.SparsityBuilder()
    sb.set_weight(
        name="0.weight",
        initial_sparsifier=sten.ScalarFractionSparsifier(0.0),
        out_format=sten.MaskedSparseTensor,
    )
    sparse_model = sb.get_sparse_model(model)

    x = torch.randn(2, 3, 10, 10, device="cuda", requires_grad=True)
    sp_x = x.detach().clone().requires_grad_(True)

    y, x_grad = amp_iteration(model, x)
    sp_y, sp_x_grad = amp_iteration(sparse_model, sp_x)

    assert torch.allclose(y, sp_y)
    assert torch.allclose(x_grad, sp_x_grad)
    assert torch.allclose(model[0].weight, sparse_model[0].weight)


if __name__ == "__main__":
    test_simple()
