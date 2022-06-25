import torch
import sten
import numpy as np
import scipy, scipy.sparse


def test_simple_graph():
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
    x = sten.SparseTensorWrapper(
        sten.DenseTensor.from_dense(torch.randn(10, 20, requires_grad=True)),
        require_grad=True,
    )
    x.grad_fmt = (
        sten.RandomFractionSparsifier(0.7),
        sten.CooTensor,
        sten.RandomFractionSparsifier(0.9),
        sten.CsrTensor,
    )
    y = sten.SparseTensorWrapper(
        sten.CsrTensor.from_dense(torch.randn(10, 20, requires_grad=True)),
        require_grad=True,
    )
    y.grad_fmt = (sten.KeepAll(), sten.DenseTensor, sten.KeepAll(), sten.DenseTensor)
    z = sparse_add(x, y)
    z.backward(
        sten.SparseTensorWrapper(sten.DenseTensor.from_dense(torch.randn(10, 20)))
    )
    assert isinstance(x.grad.wrapped_tensor, sten.CsrTensor)
    assert isinstance(y.grad.wrapped_tensor, sten.DenseTensor)


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
        dense_weight = sten.random_mask_sparsify(
            torch.randn(output_features, input_features), frac=weight_sparsity
        )
        self.weight = sten.SparseParameterWrapper(
            sten.CscTensor.from_dense(dense_weight)
        )
        self.weight.grad_fmt = (
            sten.KeepAll(),
            torch.Tensor,
            sten.RandomFractionSparsifier(self.weight_sparsity),
            sten.CscTensor,
        )
        self.bias = torch.nn.Parameter(torch.rand(output_features))
        self.bias.grad_fmt = (
            sten.KeepAll(),
            torch.Tensor,
            sten.KeepAll(),
            torch.Tensor,
        )

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


def test_build_mlp_from_scratch():
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
    model = torch.hub.load(
        "huggingface/pytorch-transformers", "model", "bert-base-uncased"
    ).encoder.layer[0]

    F = 768  # features
    B = 8  # batch
    S = 128  # seqence
    input_shape = (B, S, F)
    input = torch.rand(input_shape, requires_grad=True)
    (output,) = model(input)

    sb = sten.SparsityBuilder(model)

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

    sparse_model = sb.get_sparse_model()

    sparse_input = input.clone().detach()
    sparse_input.requires_grad = True
    (sparse_output,) = sparse_model(sparse_input)

    assert torch.allclose(output, sparse_output, atol=1e-4, rtol=1e-2)

    grad_output = torch.randn_like(sparse_output)
    sparse_output.backward(grad_output)
    output.backward(grad_output)

    assert torch.allclose(input.grad, sparse_input.grad, atol=1e-4, rtol=1e-2)


# ================ BERT Encoder ================


if __name__ == "__main__":
    test_simple_graph()
    test_build_mlp_from_scratch()
    test_modify_bert_encoder()
