import sten
from sten import SparseTensorWrapper, SparseParameterWrapper, DenseTensor
import torch
import pytest
import os


def test_add():
    shape = (3, 3)
    dx = torch.full(shape, 2.0, requires_grad=True)
    dx_copy = dx.clone().detach().requires_grad_()
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(dx_copy),
        dx_copy,
    )
    y = torch.full(shape, 3.0)
    assert torch.allclose(torch.add(dx, y), torch.add(sx, y))
    assert torch.allclose(torch.add(y, dx), torch.add(y, sx))

    grad_z = torch.full(shape, 5.0)
    z1 = torch.add(sx, y)
    z1.backward(grad_z)
    z2 = torch.add(dx, y)
    z2.backward(grad_z)
    assert torch.allclose(dx.grad, sx.grad)


def test_add_():
    shape = (3, 3)
    dx = torch.full(shape, 2.0)
    dx_copy = dx.clone().detach()
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(dx_copy),
        dx_copy,
    )
    y = torch.full(shape, 3.0)
    dx1 = dx.add_(y)
    sx1 = sx.add_(y)
    assert id(dx) == id(dx1)
    assert id(sx) == id(sx1)
    assert torch.allclose(dx, torch.full(shape, 5.0))
    assert torch.allclose(sx, torch.full(shape, 5.0))


def test_clone():
    shape = (3, 3)
    dx = torch.full(shape, 2.0, requires_grad=True)
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(dx),
        dx,
    )
    sxc = torch.clone(sx)
    # TODO: figure out how to support torch.Tensor.clone as well
    grad_x = torch.full(shape, 3.0)
    sxc.backward(grad_x)
    assert torch.allclose(sx.grad, grad_x)


def test_detach():
    shape = (3, 3)
    dx = torch.full(shape, 2.0, requires_grad=True)
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(dx),
        dx,
    )
    sxd = sx.detach()
    assert sxd.requires_grad == False
    sxd.requires_grad = True
    grad_x = torch.full(shape, 3.0)
    s_grad_x = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(grad_x),
        grad_x,
    )
    sxd.backward(s_grad_x)
    assert sx.grad is None
    assert sxd.wrapped_tensor == sx.wrapped_tensor


def test_std_mean():
    dx = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    dstd, dmean = torch.std_mean(dx, dim=1)
    dout = dstd / dmean
    dout.backward(torch.ones_like(dout))

    dx_copy = dx.detach().clone().requires_grad_(True)
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(dx_copy),
        dx_copy,
    )
    sstd, smean = torch.std_mean(sx, dim=1)
    sout = sstd / smean
    sout.backward(torch.ones_like(sout))

    assert torch.allclose(dout, sout)
    assert torch.allclose(dx.grad, sx.grad)


def test_stack():
    shape = (2, 3)
    dtensors = [
        torch.full(shape, i, dtype=torch.float32, requires_grad=True) for i in range(5)
    ]
    doutput = torch.stack(dtensors, dim=1)
    doutput.backward(torch.ones_like(doutput))

    stensors = [
        SparseTensorWrapper.wrapped_from_dense(
            DenseTensor(t.detach().clone().requires_grad_()),
            t,
        )
        for t in dtensors
    ]
    soutput = torch.stack(stensors, dim=1)
    soutput.backward(torch.ones_like(soutput))

    for dt, st in zip(dtensors, stensors):
        assert torch.allclose(dt.grad, st.grad)


def test__hash__():
    s = set()

    dx = torch.tensor([1.0, 2.0])
    sx = [
        SparseTensorWrapper.wrapped_from_dense(
            DenseTensor(dx.clone().detach()),
            dx,
        )
        for _ in range(3)
    ]

    s.add(sx[1])
    s.add(sx[2])
    s.add(sx[0])
    s.add(sx[1])
    assert len(s) == 3


def test__eq__():
    dx = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 2.0]),
    ]
    sx = [
        SparseTensorWrapper.wrapped_from_dense(
            DenseTensor(x.clone().detach()),
            x,
        )
        for x in dx
    ]

    dd = dx[0] == dx[1]
    sd1 = dx[0] == sx[1]
    sd2 = sx[0] == dx[1]

    assert torch.equal(sd1, dd)
    assert torch.equal(sd2, dd)


def test__repr__():
    x = torch.tensor([1.0, 2.0])
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(x.clone().detach()),
        x,
    )
    assert "SparseTensorWrapper" in repr(sx)

    px = SparseParameterWrapper(sx)
    rpx = repr(px)
    assert "SparseTensorWrapper" in rpx and "SparseParameterWrapper" in rpx


class MyDenseTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data

    @property
    def ndim(self):
        return "ndim_ok"

    def size(self):
        return "size_ok"

    @property
    def shape(self):
        return "shape_ok"


def test_sizes():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(x.clone().detach()),
        x,
    )
    msx = SparseTensorWrapper.wrapped_from_dense(
        MyDenseTensor(x.clone().detach()),
        x,
    )

    assert sx.ndim == x.ndim
    assert sx.size() == x.size()
    assert sx.shape == x.shape

    assert msx.ndim == "ndim_ok"
    assert msx.size() == "size_ok"
    assert msx.shape == "shape_ok"


def test_scalar_mul():
    shape = (3, 3)
    dx = torch.full(shape, 2.0, requires_grad=True)
    sx = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(dx),
        dx,
    )
    assert torch.allclose(5 * sx, sx * 5)


def test_ones_like():
    shape = (3, 3)
    dx = torch.full(shape, 2.0, requires_grad=True)
    sparsifier = sten.ScalarFractionSparsifier(0.5)
    sx = SparseTensorWrapper.wrapped_from_dense(
        sten.MaskedSparseTensor(
            sten.scalar_mask_sparsify(dx, sparsifier.fraction),
            sparsifier,
        ),
        dx,
        None,
    )
    assert torch.allclose(torch.ones_like(sx), torch.ones_like(dx))


def test_to():
    shape = (3, 3)
    dx = torch.full(shape, 2.0, requires_grad=True, dtype=torch.float32)
    sparsifier = sten.ScalarFractionSparsifier(0.5)
    sx = SparseTensorWrapper.wrapped_from_dense(
        sten.MaskedSparseTensor(
            sten.scalar_mask_sparsify(dx, sparsifier.fraction),
            sparsifier,
        ),
        dx,
        None,
    )

    assert sx.device == torch.device("cpu")
    assert sx.dtype == torch.float32
    assert sx.wrapped_tensor.data.device == torch.device("cpu")
    assert sx.wrapped_tensor.data.dtype == torch.float32

    # Signature:
    # tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
    sx64 = sx.to(torch.float64)
    assert sx64.dtype == torch.float64
    assert sx64.wrapped_tensor.data.dtype == torch.float64

    if torch.cuda.device_count() == 0:
        if "PYTEST_CURRENT_TEST" in os.environ:
            pytest.skip("No CUDA-capable device found")
        else:
            return

    # Signature:
    # tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
    sx_cuda = sx.to("cuda")
    assert sx_cuda.device.type == "cuda"
    assert sx_cuda.wrapped_tensor.data.device.type == "cuda"

    # Signature:
    # tensor.to(other, non_blocking=False, copy=False)

    other = torch.full(
        shape, 2.0, requires_grad=True, device=torch.device("cuda"), dtype=torch.float64
    )
    sx_other = sx.to(other)

    assert sx_other.dtype == torch.float64
    assert sx_other.wrapped_tensor.data.dtype == torch.float64
    assert sx_other.device.type == "cuda"
    assert sx_other.wrapped_tensor.data.device.type == "cuda"

    # check correctess of copy semantics
    sx_same = sx.to(dx)
    assert id(sx) == id(sx_same)
    sx_copy = sx.to(dx, copy=True)
    assert id(sx) != id(sx_copy)
    assert id(sx.wrapped_tensor) != id(sx_copy.wrapped_tensor)
    assert id(sx.wrapped_tensor.data) != id(sx_copy.wrapped_tensor.data)


if __name__ == "__main__":
    test_add()
    test_add_()
    test_clone()
    test_detach()
    test_std_mean()
    test_stack()
    test__hash__()
    test__eq__()
    test__repr__()
    test_sizes()
    test_scalar_mul()
    test_ones_like()
    test_to()
