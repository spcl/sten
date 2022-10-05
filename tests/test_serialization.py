import torch
import io
import sten
import scipy


def test_tensor_serialization():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    grad_fmt = (
        sten.KeepAll(),
        torch.Tensor,
        sten.RandomFractionSparsifier(0.5),
        sten.CscTensor,
    )
    sx = sten.random_fraction_sparsifier_dense_csc(
        sten.RandomFractionSparsifier(0.5),
        x,
        grad_fmt=(
            sten.KeepAll(),
            torch.Tensor,
            sten.RandomFractionSparsifier(0.5),
            sten.CscTensor,
        ),
    )
    fp = io.BytesIO()
    torch.save(sx, fp)
    fp.seek(0)
    lsx = torch.load(fp)
    assert type(lsx) == sten.SparseTensorWrapper
    assert type(lsx.wrapped_tensor) == sten.CscTensor
    assert type(lsx.wrapped_tensor.data) == scipy.sparse.csc_matrix
    assert lsx.grad_fmt == grad_fmt


def test_module_weights_serialization():
    sten.set_dispatch_failure("warn")

    model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
    sb = sten.SparsityBuilder()
    sb.set_weight(
        name="linear1.weight",
        initial_sparsifier=sten.ScalarFractionSparsifier(0.5),
        out_format=sten.CooTensor,
    )
    sparse_model = sb.get_sparse_model(model)
    assert isinstance(sparse_model.linear1.weight, sten.SparseParameterWrapper)
    assert isinstance(sparse_model.linear1.weight.wrapped_tensor, sten.CooTensor)

    fp = io.BytesIO()
    torch.save(sparse_model, fp)
    fp.seek(0)
    l_sparse_model = torch.load(fp)

    assert isinstance(l_sparse_model.linear1.weight, sten.SparseParameterWrapper)
    assert isinstance(l_sparse_model.linear1.weight.wrapped_tensor, sten.CooTensor)


if __name__ == "__main__":
    test_tensor_serialization()
    test_module_weights_serialization()
