import torch
import sten
import itertools


def test_bert_inference():
    model = torch.hub.load(
        "huggingface/pytorch-transformers", "model", "bert-base-uncased"
    )
    input = torch.randint(low=0, high=100, size=(8, 512))

    weights_to_sparsify = [
        module_name + ".weight"
        for module_name, module in model.named_modules()
        if (
            isinstance(module, torch.nn.modules.linear.Linear)
            and "encoder.layer" in module_name
        )
    ]
    assert weights_to_sparsify
    sb = sten.SparsityBuilder()
    for weight in weights_to_sparsify:
        sb.set_weight(
            name=weight,
            initial_sparsifier=sten.GroupedNMSparsifier(n=3, m=6, g=4),
            out_format=sten.GroupedNMTensor,
        )
    sparse_model = sb.get_sparse_model(model)

    output = sparse_model(input)


def test_dense_nm_conversion():
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(torch.cuda.device_count() - 1))
    else:
        device = torch.device("cpu")

    torch.manual_seed(123)
    dims = list(reversed([1, 5, 17]))
    for shape in itertools.product(dims, dims):
        shape = (3, *shape)
        base_layout = "abc"
        for layout in itertools.permutations(base_layout):
            layout = "".join(layout)
            dense_ten = torch.einsum(
                f"{base_layout}->{layout}", torch.rand(shape, device=device)
            )
            sparse_dim = layout.index("c")
            group_dim = layout.index("b")

            n = 5
            m = 12
            g = 2

            nm_ten = sten.GroupedNMTensor.from_dense(
                dense_ten,
                n=n,
                m=m,
                sparse_dim=sparse_dim,
                group_size=g,
                group_dim=group_dim,
            )

            sparsified_dense = nm_ten.to_dense()

            assert sten.grouped_nm.grouped_nm_tensor.is_correct_nm(
                dense_ten, sparsified_dense, sparse_dim=sparse_dim, n=n, m=m
            )
            preserved_magnitude = sparsified_dense.abs().sum() / dense_ten.abs().sum()
            assert preserved_magnitude > n / m  # it should be better than random

            perfect_nm_ten = sten.PerfectNMTensor.from_dense(
                dense_ten, n=n, m=m, sparse_dim=sparse_dim
            )
            perfect_sparsified_dense = perfect_nm_ten.to_dense()
            assert sten.grouped_nm.grouped_nm_tensor.is_correct_nm(
                dense_ten, perfect_sparsified_dense, sparse_dim=sparse_dim, n=n, m=m
            )
            perfect_preserved_magnitude = (
                perfect_sparsified_dense.abs().sum() / dense_ten.abs().sum()
            )
            assert perfect_preserved_magnitude >= preserved_magnitude - 1e-5
            print(
                f"shape {list(dense_ten.shape)} magnitude {n / m:.3f} < {preserved_magnitude:.3f} < {perfect_preserved_magnitude:.3f} ok"
            )


if __name__ == "__main__":
    test_bert_inference()
    test_dense_nm_conversion()
