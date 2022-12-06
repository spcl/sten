import torch
import sten
import operator
import copy
import os
import gc
import socket


class FixedMaskTensor:
    def __init__(self, val, mask):
        assert torch.all(
            torch.isclose(mask, torch.zeros_like(mask))
            | torch.isclose(mask, torch.ones_like(mask))
        )
        self.val = val
        self.mask = mask

    def to_dense(self):
        return copy.deepcopy(self.val.data)


@sten.register_sparsifier_implementation(
    sparsifier=sten.ScalarFractionSparsifier, inp=torch.Tensor, out=FixedMaskTensor
)
def dense_to_fixed(sparsifier, tensor, grad_fmt=None):
    num_zeros = int(sparsifier.fraction * tensor.numel())
    # double argsort to get the inverted mapping
    indices = torch.argsort(torch.argsort(tensor.flatten().abs())).reshape(tensor.shape)
    mask = torch.where(
        indices < num_zeros,
        torch.zeros_like(tensor, dtype=torch.bool),
        torch.ones_like(tensor, dtype=torch.bool),
    )
    assert torch.sum(mask) == tensor.numel() - num_zeros
    val = (tensor * mask).detach().clone()
    return sten.SparseTensorWrapper.wrapped_from_dense(
        FixedMaskTensor(val, mask),
        tensor,
        grad_fmt,
    )


@sten.register_sparsifier_implementation(
    sparsifier=sten.SameFormatSparsifier, inp=torch.Tensor, out=FixedMaskTensor
)
def same_to_fixed(sparsifier, tensor, grad_fmt=None):
    ref = sparsifier.ref_sp_ten.wrapped_tensor
    num_zeros = tensor.numel() - torch.sum(ref.mask)
    # double argsort to get the inverted mapping
    indices = torch.argsort(torch.argsort(tensor.flatten().abs())).reshape(tensor.shape)
    mask = torch.where(
        indices < num_zeros,
        torch.zeros_like(tensor, dtype=torch.bool),
        torch.ones_like(tensor, dtype=torch.bool),
    )
    assert torch.sum(mask) == tensor.numel() - num_zeros
    val = (tensor * mask).detach().clone()
    assert torch.allclose(val * mask, val)
    return sten.SparseTensorWrapper.wrapped_from_dense(
        FixedMaskTensor(val, mask),
        sparsifier.ref_sp_ten,
        grad_fmt,
    )


def sparse_ddp_all_reduce_hook(state, bucket):
    dense_buf = bucket.buffer()

    total_elems = 0
    for p in bucket.parameters():
        if isinstance(p, sten.SparseTensorWrapper):
            total_elems += p.numel()

    # reduce all sparse tensors
    sparse_buf = torch.zeros(
        max(total_elems, 1), dtype=dense_buf.dtype, device=dense_buf.device
    )
    processed = 0
    for p in bucket.parameters():
        if isinstance(p, sten.SparseTensorWrapper):
            sparse_buf[
                processed : processed + p.numel()
            ] = p.grad.wrapped_tensor.to_dense().flatten()
            processed += p.numel()
    assert processed == total_elems

    sparse_buf /= torch.distributed.get_world_size()
    fut_sparse = torch.distributed.all_reduce(
        sparse_buf, op=torch.distributed.ReduceOp.SUM, async_op=True
    ).get_future()

    fut_sparse.wait()

    check_the_same(fut_sparse.value()[0])

    # reduce all dense tensors
    dense_buf /= torch.distributed.get_world_size()
    fut_dense = torch.distributed.all_reduce(
        dense_buf, op=torch.distributed.ReduceOp.SUM, async_op=True
    ).get_future()

    fut_dense.wait()

    check_the_same(fut_dense.value()[0])

    fut = torch.futures.collect_all([fut_sparse, fut_dense])

    def postporcess(fut):
        spase_fut, dense_fut = fut.value()
        [sparse_buf] = spase_fut.value()
        [dense_buf] = dense_fut.value()
        # process sparse gradients maqnually
        processed = 0
        for p in bucket.parameters():
            if isinstance(p, sten.SparseTensorWrapper):
                dense_grad = sparse_buf[processed : processed + p.numel()].reshape(
                    p.shape
                )
                sparsifier = sten.get_sparsifier_implementation(
                    sten.SameFormatSparsifier,
                    torch.Tensor,
                    p.grad.wrapped_tensor.__class__,
                )
                check_the_same(dense_grad)
                reduced_sparse_grad = sparsifier(
                    sten.SameFormatSparsifier(p.grad), dense_grad
                )
                p.grad.init_from_other(reduced_sparse_grad)
                processed += p.numel()
        assert processed == total_elems
        # return dense_buf as is, it will be used to update grad values of dense tensors by DDP
        check_the_same(bucket.buffer())
        return dense_buf

    return fut.then(postporcess)


def check_the_same(tensor):
    tt = copy.deepcopy(tensor)
    ttl = [torch.rand_like(tt) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(ttl, tt)
    for t in ttl:
        assert torch.all(tt == t)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(3, 5)
        self.l1 = torch.nn.Linear(5, 7)

    def forward(self, input, target):
        return torch.nn.functional.smooth_l1_loss(self.l1(self.l0(input)), target)


def tensors_in_use():
    gc.collect()
    return len([obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)])


def linear_layers_with_ddp(rank, world_size):
    torch.distributed.init_process_group(
        backend="gloo", world_size=world_size, rank=rank
    )

    model = MyModule()

    sparse_weight_path = "l0.weight"
    dense_weight_path = "l1.weight"

    sb = sten.SparsityBuilder()
    sparsifier = sten.ScalarFractionSparsifier(0.7)
    sb.set_weight(
        name=sparse_weight_path,
        initial_sparsifier=sparsifier,
        out_format=FixedMaskTensor,
    )
    sb.set_weight_grad(
        name=sparse_weight_path,
        external_sparsifier=sparsifier,
        out_format=FixedMaskTensor,
    )

    sparse_model = sb.sparsify_model_inplace(model)

    assert sparse_model.l0.weight.requires_grad

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        sparse_model,
    )
    ddp_model.register_comm_hook(state=None, hook=sparse_ddp_all_reduce_hook)

    assert ddp_model.module.l0.weight.requires_grad

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=0.01)

    last_tensors_in_use = None

    num_steps = 10
    for step in range(num_steps):
        # +++ check old dense tensors
        ns_old_weight = operator.attrgetter(dense_weight_path)(ddp_model.module)
        assert not isinstance(ns_old_weight, sten.SparseTensorWrapper)
        dense_ns_old_weight = copy.deepcopy(ns_old_weight)
        check_the_same(dense_ns_old_weight)
        # === check old dense tensors

        # +++ check old sparse tensors
        old_weight = operator.attrgetter(sparse_weight_path)(ddp_model.module)
        old_wrapped_tensor = old_weight
        assert isinstance(old_weight, sten.SparseTensorWrapper)
        dense_old_weight = old_weight.wrapped_tensor.to_dense()
        check_the_same(dense_old_weight)
        # === check old sparse tensors

        input = torch.randn(2, 3)
        target = torch.randn(2, 7)
        loss = ddp_model(input, target)

        assert ddp_model.module.l0.weight.requires_grad

        loss.backward()

        assert ddp_model.module.l0.weight.requires_grad

        # +++ check dense grad
        ns_new_weight = operator.attrgetter(dense_weight_path)(ddp_model.module)
        check_the_same(ns_new_weight.grad)
        # === check dense grad

        # +++ check sparse grad
        new_weight = operator.attrgetter(sparse_weight_path)(ddp_model.module)
        check_the_same(new_weight.grad.wrapped_tensor.to_dense())
        # === check sparse grad

        optimizer.step()

        assert ddp_model.module.l0.weight.requires_grad
        optimizer.zero_grad(set_to_none=True)
        assert ddp_model.module.l0.weight.requires_grad

        # +++ check dense weight
        ns_new_weight = operator.attrgetter(dense_weight_path)(ddp_model.module)
        assert not isinstance(ns_new_weight, sten.SparseTensorWrapper)
        dense_ns_new_weight = copy.deepcopy(ns_new_weight)
        assert not torch.all(dense_ns_old_weight == dense_ns_new_weight)
        check_the_same(dense_ns_new_weight)
        # === check dense weight

        # +++ check sparse weight
        new_weight = operator.attrgetter(sparse_weight_path)(ddp_model.module)
        new_wrapped_tensor = new_weight.wrapped_tensor
        assert isinstance(new_weight, sten.SparseTensorWrapper)
        dense_new_weight = new_weight.wrapped_tensor.to_dense()
        assert old_wrapped_tensor is not new_wrapped_tensor
        assert not torch.all(dense_old_weight == dense_new_weight)
        check_the_same(dense_new_weight)
        # === check sparse weight

        optimizer.zero_grad(set_to_none=True)

        # +++ check dense grad
        ns_new_weight = operator.attrgetter(dense_weight_path)(ddp_model.module)
        assert ns_new_weight.grad is None
        # === check dense grad

        # +++ check sparse grad
        new_weight = operator.attrgetter(sparse_weight_path)(ddp_model.module)
        assert new_weight.grad is None
        # === check sparse grad

        # check for memory leaks
        if last_tensors_in_use is None:
            last_tensors_in_use = tensors_in_use()
        else:
            assert last_tensors_in_use == tensors_in_use()

    torch.distributed.destroy_process_group()


def get_free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_ddp():
    port = get_free_port()

    world_size = 2
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.multiprocessing.spawn(
        linear_layers_with_ddp, args=(world_size,), nprocs=world_size
    )


if __name__ == "__main__":
    test_ddp()
