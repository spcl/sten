import torch
import sten
import numpy as np
import scipy, scipy.sparse
import copy


def test_simple_graph():
    sten.set_dispatch_failure("raise")
    sparse_add = sten.sparsified_op(
        orig_op=torch.add,
        out_fmt=tuple(
            [
                (
                    sten.RandomFractionSparsifier(0.6),
                    sten.CsrTensor,
                    sten.KeepAll(),
                    sten.CsrTensor,
                ),
            ]
        ),
        grad_out_fmt=(
            [
                (sten.KeepAll(), torch.Tensor, sten.KeepAll(), sten.DenseTensor),
            ]
        ),
    )
    dense_x = torch.randn(10, 20, requires_grad=True)
    x = sten.torch_tensor_to_wrapped_dense(
        sten.KeepAll(),
        dense_x,
        (
            sten.RandomFractionSparsifier(0.7),
            sten.CooTensor,
            sten.RandomFractionSparsifier(0.9),
            sten.CsrTensor,
        ),
    )
    dense_y = torch.randn(10, 20, requires_grad=True)
    y = sten.torch_tensor_to_csr(
        sten.KeepAll(),
        dense_y,
        (sten.KeepAll(), sten.DenseTensor, sten.KeepAll(), sten.DenseTensor),
    )
    z = sparse_add(x, y)
    dense_grad_z = torch.randn(10, 20)
    grad_z = sten.torch_tensor_to_wrapped_dense(
        sten.KeepAll(),
        dense_grad_z,
    )
    z.backward(grad_z)
    assert isinstance(x.grad.wrapped_tensor, sten.CsrTensor)
    assert isinstance(y.grad.wrapped_tensor, sten.DenseTensor)


def test_modify_transformer_encoder_layer():
    sten.set_dispatch_failure("warn")

    model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
    sb = sten.SparsityBuilder()
    sb.set_weight(
        name="linear1.weight",
        initial_sparsifier=sten.ScalarFractionSparsifier(0.5),
        out_format=sten.CooTensor,
    )
    sb.set_interm(
        name="relu",
        external_sparsifier=sten.ScalarFractionSparsifier(0.5),
        out_format=sten.CooTensor,
    )
    sparse_model = sb.get_sparse_model(model)
    assert isinstance(sparse_model.linear1.weight, sten.SparseParameterWrapper)
    assert isinstance(sparse_model.linear1.weight.wrapped_tensor, sten.CooTensor)
    sparse_model(torch.randn(8, 128, 512))


# ++++++++++++++++ MLP ++++++++++++++++


class MLP(torch.nn.Module):
    def __init__(self, channel_sizes):
        super().__init__()
        self.layers = torch.nn.Sequential()
        in_out_pairs = list(zip(channel_sizes[:-1], channel_sizes[1:]))
        for idx, (in_channels, out_channels) in enumerate(in_out_pairs):
            if idx != 0:
                self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, input):
        return self.layers(input)


class SparseLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, weight_sparsity):
        super().__init__()
        self.weight_sparsity = weight_sparsity
        self.weight = sten.SparseParameterWrapper(
            sten.random_fraction_sparsifier_dense_csc(
                sten.RandomFractionSparsifier(self.weight_sparsity),
                torch.randn(output_features, input_features),
                (
                    sten.KeepAll(),
                    torch.Tensor,
                    sten.RandomFractionSparsifier(self.weight_sparsity),
                    sten.CscTensor,
                ),
            )
        )
        self.bias = torch.nn.Parameter(torch.rand(output_features))

    def forward(self, input):
        sparse_op = sten.sparsified_op(
            orig_op=torch.nn.functional.linear,
            out_fmt=tuple(
                [(sten.KeepAll(), torch.Tensor, sten.KeepAll(), torch.Tensor)]
            ),
            grad_out_fmt=tuple(
                [(sten.KeepAll(), torch.Tensor, sten.KeepAll(), torch.Tensor)]
            ),
        )
        return sparse_op(input, self.weight, self.bias)


class SparseMLP(torch.nn.Module):
    def __init__(self, channel_sizes, weight_sparsity):
        super().__init__()
        self.layers = torch.nn.Sequential()
        in_out_pairs = list(zip(channel_sizes[:-1], channel_sizes[1:]))
        for idx, (in_channels, out_channels) in enumerate(in_out_pairs):
            if idx != 0:
                self.layers.append(torch.nn.ReLU())
            self.layers.append(SparseLinear(in_channels, out_channels, weight_sparsity))

    def forward(self, input):
        return self.layers(input)


@sten.register_fwd_op_impl(
    operator=torch.allclose,
    inp=[torch.Tensor, sten.sten.CscTensor],
    out=[],
)
def allclose_impl(ctx, inputs, output_sparsifiers):
    input1, input2 = inputs
    return torch.allclose(input1, input2.wrapped_tensor.to_dense())


def test_build_mlp_from_scratch():
    sten.set_dispatch_failure("raise")
    input = torch.randn(15, 50)
    channel_sizes = [50, 40, 30, 20, 30, 10]

    # run dense
    model = MLP(channel_sizes)
    output = model(input)

    # run sparse
    sparse_model = SparseMLP(channel_sizes, weight_sparsity=0.7)
    sparse_output = sparse_model(input)

    # compare sparse and dense results
    sparse_model = SparseMLP(channel_sizes, weight_sparsity=0)
    sparse_model.load_state_dict(model.state_dict())
    sparse_output = sparse_model(input)
    assert torch.allclose(output, sparse_output, atol=1e-4, rtol=1e-2)

    # compare sparse amd demse gradoemts
    dense_input = torch.randn(15, 50, requires_grad=True)
    sparse_input = dense_input.detach().clone()
    sparse_input.requires_grad = True
    dense_output = model(dense_input)
    sparse_output = sparse_model(sparse_input)

    out_grad = torch.rand_like(sparse_output)
    dense_output.backward(out_grad)
    sparse_output.backward(out_grad)

    assert torch.allclose(dense_input.grad, sparse_input.grad)
    for k in model.state_dict():
        dv = model.state_dict()[k]
        sv = sparse_model.state_dict()[k]
        assert torch.allclose(dv, sv)


# ================ MLP ================

# ++++++++++++++++ BERT Encoder ++++++++++++++++


def test_modify_bert_encoder():
    sten.set_dispatch_failure("raise")
    model = torch.hub.load(
        "huggingface/pytorch-transformers", "model", "bert-base-uncased"
    ).encoder.layer[0]

    F = 768  # features
    B = 8  # batch
    S = 128  # seqence
    input_shape = (B, S, F)
    input = torch.rand(input_shape, requires_grad=True)
    (output,) = model(input)

    sb = sten.SparsityBuilder()

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.linear.Linear):
            sb.set_weight(
                module_name + ".weight",
                sten.ScalarFractionSparsifier(0),
                sten.KeepAll(),
                torch.Tensor,
                sten.KeepAll(),
                sten.CsrTensor,
            )

    sb.set_interm(
        "intermediate.gelu",
        sten.RandomFractionSparsifier(0),
        sten.CooTensor,
        sten.KeepAll(),
        sten.CooTensor,
    )

    sparse_model = sb.get_sparse_model(model)

    sparse_input = input.clone().detach()
    sparse_input.requires_grad = True
    (sparse_output,) = sparse_model(sparse_input)

    assert torch.allclose(output, sparse_output, atol=1e-4, rtol=1e-2)

    grad_output = torch.randn_like(sparse_output)
    sparse_output.backward(grad_output)
    output.backward(grad_output)

    assert torch.allclose(input.grad, sparse_input.grad, atol=1e-4, rtol=1e-2)


# ================ BERT Encoder ================

# ++++++++++++++++ Custom implementations ++++++++++++++++


class MyRandomFractionSparsifier:
    def __init__(self, fraction):
        self.fraction = fraction


class MyCscTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return torch.from_numpy(self.data.todense())


@sten.register_sparsifier_implementation(
    sparsifier=MyRandomFractionSparsifier, inp=torch.Tensor, out=MyCscTensor
)
def scalar_fraction_sparsifier_dense_coo(sparsifier, tensor):
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCscTensor(
            scipy.sparse.csc_matrix(
                sten.random_mask_sparsify(tensor, frac=sparsifier.fraction)
            )
        ),
        tensor,
    )


@sten.register_fwd_op_impl(
    operator=torch.mm,
    inp=(MyCscTensor, torch.Tensor),
    out=tuple([(sten.KeepAll, torch.Tensor)]),
)
def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
    input1, input2 = inputs
    ctx.save_for_backward(input1, input2)
    output = torch.from_numpy(input1.wrapped_tensor.data @ input2.numpy())
    return output


@sten.register_bwd_op_impl(
    operator=torch.mm,
    grad_out=(torch.Tensor,),
    grad_inp=(
        (sten.KeepAll, torch.Tensor),
        (sten.KeepAll, torch.Tensor),
    ),
    inp=(MyCscTensor, torch.Tensor),
)
def torch_mm_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input1, input2 = ctx.saved_tensors
    [grad_output] = grad_outputs
    grad_input1 = torch.mm(grad_output, input2.T)
    grad_input2 = torch.from_numpy(input1.wrapped_tensor.data.transpose() @ grad_output)
    return grad_input1, grad_input2


@sten.register_bwd_op_impl(
    operator=torch.add,
    grad_out=[MyCscTensor],
    grad_inp=(
        (sten.KeepAll, torch.Tensor),
        (sten.KeepAll, torch.Tensor),
    ),
    inp=(torch.Tensor, torch.Tensor),
)
def torch_add_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    [grad_output] = grad_outputs
    dense_output = torch.from_numpy(grad_output.wrapped_tensor.data.todense())
    return dense_output.clone().detach(), dense_output.clone().detach()


class MyRandomFractionSparsifierFallback:
    def __init__(self, fraction):
        self.fraction = fraction


class MyCscTensorFallback:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return torch.from_numpy(self.data.todense())


@sten.register_sparsifier_implementation(
    sparsifier=MyRandomFractionSparsifierFallback,
    inp=torch.Tensor,
    out=MyCscTensorFallback,
)
def my_default_sparsifier(sparsifier, tensor, grad_fmt=None):
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyCscTensorFallback(
            scipy.sparse.csc_matrix(
                sten.random_mask_sparsify(tensor, frac=sparsifier.fraction)
            )
        ),
        tensor,
        grad_fmt,
    )


def add_mm_implementations(sparsifier_cls, tensor_cls):
    a = torch.randn(10, 20, requires_grad=True)
    b = torch.randn(10, 20, requires_grad=True)
    c = torch.randn(20, 30, requires_grad=True)
    grad_d = torch.randn(10, 30)

    d = torch.mm(torch.add(a, b), c)
    d.backward(grad_d)

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert c.grad.shape == c.shape

    grad_a = a.grad
    grad_b = b.grad
    grad_c = c.grad

    sparse_add = sten.sparsified_op(
        orig_op=torch.add,
        out_fmt=tuple(
            [
                (
                    sten.KeepAll(),
                    torch.Tensor,
                    sparsifier_cls(0),
                    tensor_cls,
                )
            ]
        ),
        grad_out_fmt=tuple(
            [
                (
                    sten.KeepAll(),
                    torch.Tensor,
                    sparsifier_cls(0),
                    tensor_cls,
                )
            ]
        ),
    )

    a.grad = None
    b.grad = None
    c.grad = None

    d2 = torch.mm(sparse_add(a, b), c)
    d2.backward(grad_d)

    assert torch.allclose(d, d2, rtol=1e-2, atol=1e-4)

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    assert c.grad.shape == c.shape

    assert torch.allclose(grad_a, a.grad, rtol=1e-2, atol=1e-4)
    assert torch.allclose(grad_b, b.grad, rtol=1e-2, atol=1e-4)
    assert torch.allclose(grad_c, c.grad, rtol=1e-2, atol=1e-4)


def test_custom_implementations():
    sten.set_dispatch_failure("raise")
    add_mm_implementations(MyRandomFractionSparsifier, MyCscTensor)


def test_fallback_implementations():
    sten.set_dispatch_failure("warn")
    add_mm_implementations(MyRandomFractionSparsifierFallback, MyCscTensorFallback)


# ================ Custom implementations ================


from test_patching import FixedMaskTensor


def test_optimizer_sparsification():
    model = torch.nn.Linear(20, 30)

    orig_weight = copy.deepcopy(model.weight)

    sb = sten.SparsityBuilder()
    sparsifier = sten.ScalarFractionSparsifier(0.0)
    sb.set_weight(
        name="weight",
        initial_sparsifier=sparsifier,
        out_format=FixedMaskTensor,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    model, optimizer = sb.sparsify_model_and_optimizer_inplace(model, optimizer)

    loss = torch.nn.L1Loss()

    x = torch.randn(13, 20)
    y = model(x)
    l = loss(y, torch.ones_like(y))
    l.backward()

    assert torch.equal(orig_weight, model.weight)

    optimizer.step()
    optimizer.zero_grad()

    assert not torch.allclose(orig_weight, model.weight)


def test_gradient_autowrapping():
    sparse_abs = sten.sparsified_op(
        orig_op=torch.abs,
        out_fmt=[
            (
                sten.KeepAll(),
                sten.torch.Tensor,
                sten.RandomFractionSparsifier(0.6),
                sten.CsrTensor,
            ),
        ],
        grad_out_fmt=[
            (sten.KeepAll(), torch.Tensor, sten.KeepAll(), torch.Tensor),
        ],
    )
    x = torch.randn(10, 20, requires_grad=True)
    z = torch.abs(sparse_abs(x))
    grad_z = torch.randn_like(z)
    z.backward(grad_z)


class MyTestSparsifier:
    pass


class MyTestFormat:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data.detach().clone()


@sten.register_sparsifier_implementation(
    sparsifier=MyTestSparsifier, inp=torch.Tensor, out=MyTestFormat
)
def dense_to_grouped_nm(sparsifier, tensor, grad_fmt=None):
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyTestFormat(tensor.detach().clone()),
        tensor,
        grad_fmt,
    )


@sten.register_sparsifier_implementation(
    sparsifier=sten.SameFormatSparsifier, inp=torch.Tensor, out=MyTestFormat
)
def dense_to_grouped_nm(sparsifier, tensor, grad_fmt=None):
    wrp = sparsifier.ref_sp_ten.wrapped_tensor
    assert isinstance(wrp, MyTestFormat)
    # for this simple test wrp is not used
    # but it may be practical for real sparse format
    return sten.SparseTensorWrapper.wrapped_from_dense(
        MyTestFormat(tensor.detach().clone()),
        tensor,
        grad_fmt,
    )


def test_grad_fmt_assignment():
    # check that optimizer step assigns gradient properly
    # without damaging grad_fmt field of original tensor
    model = torch.nn.Linear(3, 5)
    sb = sten.SparsityBuilder()
    sb.set_weight(
        name="weight",
        initial_sparsifier=MyTestSparsifier(),
        out_format=MyTestFormat,
    )
    sb.set_weight_grad(
        name="weight",
        inline_sparsifier=MyTestSparsifier(),
        tmp_format=MyTestFormat,
        external_sparsifier=sten.KeepAll(),
        out_format=MyTestFormat,
    )
    model = sb.sparsify_model_inplace(model)
    optim = torch.optim.AdamW(model.parameters())

    def check_grad_fmt(grad_fmt):
        assert grad_fmt[0].__class__ == MyTestSparsifier
        assert grad_fmt[1] == MyTestFormat
        assert grad_fmt[2].__class__ == sten.KeepAll
        assert grad_fmt[3] == MyTestFormat

    check_grad_fmt(model.weight.grad_fmt)
    assert model.weight.requires_grad

    steps = 10
    for _ in range(steps):
        inp = torch.randn(2, 3)
        outputs = model(inp)
        loss = torch.sum(outputs)
        loss.backward()
        optim.step()
        optim.zero_grad()

        check_grad_fmt(model.weight.grad_fmt)
        assert model.weight.requires_grad


if __name__ == "__main__":
    test_simple_graph()
    test_modify_transformer_encoder_layer()
    test_build_mlp_from_scratch()
    test_modify_bert_encoder()
    test_custom_implementations()
    test_fallback_implementations()
    test_optimizer_sparsification()
    test_gradient_autowrapping()
    test_grad_fmt_assignment()
