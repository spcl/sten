import torch
import copy
import numpy as np
import torch.fx
import inspect
import scipy, scipy.sparse
import logging
import itertools


# ++++++++++++++++++++++ Core implementation ++++++++++++++++++++++

DISPATCH_RAISE = "raise"
DISPATCH_WARN = "warn"

DISPATCH_FAILURE = DISPATCH_WARN


def set_dispatch_failure(action):
    # 'warn' or 'raise'
    if action not in (DISPATCH_RAISE, DISPATCH_WARN):
        raise ValueError("Unknown action")
    global DISPATCH_FAILURE
    DISPATCH_FAILURE = action


class DispatchError(Exception):
    pass


HANDLED_FUNCTIONS = {}


def implements(*torch_functions):
    def decorator(implementation):
        for tf in torch_functions:
            HANDLED_FUNCTIONS[tf] = implementation
        return implementation

    return decorator


class SparseTensorWrapper(torch.Tensor):
    # When autograd runs, it casts gradient from SparseTensorWrapper to torch.Tensor.
    # To fix it, we manually assign SparseTensorWrapper to .grad field
    def grad_fix_hook(self, grad):
        self.grad = grad

    def update_grad_fix_hook(self):
        if self.requires_grad and (not hasattr(self, "grad_fix_hook_handle")):
            self.grad_fix_hook_handle = self.register_hook(self.grad_fix_hook)
        elif (not self.requires_grad) and hasattr(self, "grad_fix_hook_handle"):
            self.grad_fix_hook_handle.remove()
            del self.grad_fix_hook_handle

    @staticmethod
    def wrapped_from_dense(wrapped, dense, grad_fmt=None):
        if grad_fmt is None:
            grad_fmt = (KeepAll(), DenseTensor, KeepAll(), DenseTensor)
        return SparseTensorWrapper(
            wrapped_tensor=wrapped,
            requires_grad=dense.requires_grad,
            grad_fmt=grad_fmt,
            dtype=dense.dtype,
            device=dense.device,
        )

    def __new__(
        cls,
        wrapped_tensor,
        requires_grad,
        grad_fmt,
        dtype,
        device,
    ):
        assert not isinstance(wrapped_tensor, SparseTensorWrapper)
        tensor = torch.Tensor._make_subclass(
            cls,
            torch.tensor([987654321.123456789], dtype=dtype, device=device),
            requires_grad,
        )
        tensor.wrapped_tensor = wrapped_tensor
        tensor.update_grad_fix_hook()
        tensor.grad_fmt = grad_fmt
        return tensor

    def init_from_other(self, other):
        assert isinstance(other, SparseTensorWrapper)
        self.wrapped_tensor = other.wrapped_tensor
        self.requires_grad = other.requires_grad
        self.grad_fmt = other.grad_fmt
        if self.dtype != other.dtype or self.device != other.device:
            raise Exception("This should never happen.")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            if func.__qualname__ in (
                "getset_descriptor.__get__",
                "getset_descriptor.__set__",
            ):
                funcself = func.__self__
            else:
                funcself = ""
            err_msg = f"SparseTensorWrapper function is not explicitly registered: {func} {funcself}."
            if DISPATCH_FAILURE == DISPATCH_WARN:
                logging.warning(f"{err_msg} Fallback to dense implementation.")
                return sparse_autoconvert_impl(
                    super().__torch_function__, func, types, *args, **kwargs
                )
            else:
                raise DispatchError(err_msg)
        return HANDLED_FUNCTIONS[func](
            super().__torch_function__, func, types, *args, **kwargs
        )


class SparseParameterWrapper(SparseTensorWrapper, torch.nn.parameter.Parameter):
    @staticmethod
    def wrapped_from_dense(wrapped_tensor, dense_tensor, grad_fmt=None):
        raise Exception(
            "Do not use this, instead call wrapped_from_dense of SparseTensorWrapper and then wrap"
        )

    def __new__(
        cls,
        sparse_tensor_wrapper,
    ):
        assert isinstance(sparse_tensor_wrapper, SparseTensorWrapper)
        return super().__new__(
            cls,
            wrapped_tensor=sparse_tensor_wrapper.wrapped_tensor,
            requires_grad=sparse_tensor_wrapper.requires_grad,
            grad_fmt=sparse_tensor_wrapper.grad_fmt,
            dtype=sparse_tensor_wrapper.dtype,
            device=sparse_tensor_wrapper.device,
        )

    @staticmethod
    def base_type(type):
        return SparseTensorWrapper if type == SparseParameterWrapper else type

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        base_types = tuple(SparseParameterWrapper.base_type(t) for t in types)
        result = SparseTensorWrapper.__torch_function__(func, base_types, args, kwargs)
        return result


@implements(
    torch.Tensor.register_hook,
    torch.Tensor.__hash__,
    torch.Tensor.__array__,
    torch.Tensor.type,
    torch.Tensor.requires_grad.__get__,
    torch.Tensor.grad.__set__,
    torch.Tensor.grad.__get__,
    torch.Tensor.is_leaf.__get__,
    torch.Tensor.grad_fn.__get__,
    torch.Tensor.dtype.__get__,
    torch.Tensor.backward,
    torch.Tensor.is_floating_point,
    torch.Tensor.device.__get__,
    torch.Tensor.is_sparse.__get__,  # Pretend to be dense to prevent complaints from PyTorch
    torch.Tensor.is_complex,
    torch.Tensor.element_size,
)
def sparse_default_impl(base_impl, func, types, *args, **kwargs):
    return base_impl(func, types, args, kwargs)


@implements(torch.Tensor.to)
def sparse_torch_tensor_to(
    base_impl,
    func,
    types,
    self,
    device=None,
    dtype=None,
    non_blocking=False,
    copy=False,
    memory_format=torch.preserve_format,
):
    if (
        (dtype is not None)
        or non_blocking
        or copy
        or (memory_format != torch.preserve_format)
    ):
        raise Exception("Not implemented")
    wrapper = SparseTensorWrapper(
        self.wrapped_tensor.to(device),
        self.requires_grad,
        self.grad_fmt,
        dtype=None,
        device=device,
    )
    return wrapper


@implements(torch.nn.functional.linear, torch.mm)
def sparse_operator_dispatch(base_impl, func, types, *args, **kwargs):
    op = sparsified_op(
        func,
        [(KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)],
        [(KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)],
    )
    return op(*args, **kwargs)


# @implements(torch.Tensor.detach)
# def sparse_copying_impl(base_impl, func, types, *args, **kwargs):
#     copied_tensor = base_impl(func, types, args, kwargs)
#     assert isinstance(copied_tensor, SparseTensorWrapper) and not hasattr(
#         copied_tensor, "wrapped_tensor"
#     )
#     original_tensor = args[0]
#     assert isinstance(original_tensor, SparseTensorWrapper) and hasattr(
#         original_tensor, "wrapped_tensor"
#     )
#     copied_tensor.wrapped_tensor = original_tensor.wrapped_tensor
#     return copied_tensor


# @implements(
#     torch.Tensor.shape.__get__,
#     torch.Tensor.data.__get__,
# )
# def sparse_redirection_property_impl(base_impl, func, types, *args, **kwargs):
#     [original_tensor] = args
#     result = getattr(original_tensor.wrapped_tensor, func.__self__.__name__)
#     if isinstance(result, type(original_tensor.wrapped_tensor)):
#         return SparseTensorWrapper(
#             result,
#             original_tensor.requires_grad,
#             original_tensor.grad_fmt,
#             dtype=original_tensor.dtype,
#             device=original_tensor.device,
#         )
#     return result

# +++++++++++++++++++++++ fallback implementations +++++++++++++++++++++++

# case 1
@implements(
    torch.Tensor.clone,
)
def sparse_functional_fallback(base_impl, func, types, *args, **kwargs):
    """
    This fallback works for functions that DO NOT MODIFY inputs and return NEW tensors
    """
    if len(kwargs) != 0 or len(args) == 1:
        raise NotImplementedError()
    original_tensor = args[0]
    result = getattr(original_tensor.wrapped_tensor, func.__name__)(
        *(args[1:]), **kwargs
    )
    if isinstance(result, type(original_tensor.wrapped_tensor)):
        return SparseTensorWrapper(
            result, original_tensor.requires_grad, original_tensor.grad_fmt
        )
    return result


# case 2
@implements(
    torch.Tensor.detach,
)
def sparse_reference_fallback(base_impl, func, types, *args, **kwargs):
    """
    This fallback works for functions that DO NOT MODIFY inputs and return tensors that SHARE something with inputs
    """
    copied_tensor = base_impl(func, types, args, kwargs)
    assert isinstance(copied_tensor, SparseTensorWrapper) and not hasattr(
        copied_tensor, "wrapped_tensor"
    )
    original_tensor = args[0]
    assert isinstance(original_tensor, SparseTensorWrapper) and hasattr(
        original_tensor, "wrapped_tensor"
    )
    copied_tensor.wrapped_tensor = original_tensor.wrapped_tensor
    return copied_tensor


# case 3
@implements(
    torch.Tensor.shape.__get__,
)
def sparse_read_only_fallback(base_impl, func, types, *args, **kwargs):
    """
    This fallback works for functions that DO NOT MODIFY inputs and DO NOT OUTPUT any tensors
    """
    d_args, d_kwargs = densify_params(args, kwargs)
    return func(*d_args, *d_kwargs)


# case 4
@implements(torch.Tensor.copy_)
def torch_modify_inplace_fallback(base_impl, func, types, *args, **kwargs):
    """
    This fallback works for functions that MODIFY inputs and DO NOT OUTPUT any tensors.
    """
    d_args, d_kwargs = densify_params(args, kwargs)
    output = func(*d_args, **d_kwargs)
    resparsify_params(args, kwargs, d_args, d_kwargs)
    return output


# ======================= fallback implementations =======================


def resparsify_params(args, kwargs, changed_args, changed_kwargs):
    def resparsify(arg, changed_arg):
        if isinstance(arg, SparseTensorWrapper):
            sparsifier = get_sparsifier_implementation(
                SameFormatSparsifier, torch.Tensor, arg.wrapped_tensor.__class__
            )
            sparse_arg = sparsifier(SameFormatSparsifier(arg), changed_arg)
            arg.init_from_other(sparse_arg)
            return arg.wrapped_tensor.to_dense()
        if isinstance(arg, list):
            return [resparsify(x) for x in arg]
        if isinstance(arg, dict):
            return {k: resparsify(v) for k, v in arg.items()}
        else:
            return arg

    for a, ca in zip(args, changed_args):
        resparsify(a, ca)
    for k, v in kwargs.items():
        resparsify(v, changed_kwargs[k])


@implements(
    torch.Tensor.size,
    torch.Tensor.numel,
    torch.norm,
    torch.Tensor.zero_,
    torch.Tensor.__reduce_ex__,
    torch.Tensor.reshape,
)
def sparse_redirection_function_impl(base_impl, func, types, *args, **kwargs):
    original_tensor = args[0]
    result = getattr(original_tensor.wrapped_tensor, func.__name__)(
        *(args[1:]), **kwargs
    )
    if isinstance(result, type(original_tensor.wrapped_tensor)):
        return SparseTensorWrapper(
            result, original_tensor.requires_grad, original_tensor.grad_fmt
        )
    return result


@implements(torch.Tensor.requires_grad_, torch.Tensor.requires_grad.__set__)
def torch_tensor_requires_grad_(base_impl, func, types, self, requries_grad=True):
    res = base_impl(func, types, (self, requries_grad))
    self.update_grad_fix_hook()
    return res


# @implements(torch.Tensor.copy_)
# def torch_tensor_copy_(base_impl, func, types, self, src, non_blocking=False):
#     if get_format(self) == torch.Tensor:  # sparse to dense
#         self.copy_(src.wrapped_tensor.to_dense())
#     elif get_format(src) == torch.Tensor:  # dense to sparse
#         self.wrapped_tensor = self.wrapped_tensor.clone_format(src)
#     else:  # sparse to sparse
#         converter = get_sparsifier_implementation(
#             KeepAll, get_format(src), get_format(self)
#         )
#         dst = converter(KeepAll(), src)
#         self.wrapped_tensor = dst.wrapped_tensor


@implements(torch.Tensor.add_)
def torch_tensor_add_(base_impl, func, types, self, other, *, alpha=1):
    if get_format(self) == torch.Tensor:  # sparse to dense
        self.add_(other.wrapped_tensor.to_dense().to(device=other.device), alpha=alpha)
    elif get_format(other) == torch.Tensor:  # dense to sparse
        self.wrapped_tensor.add_(other, alpha=alpha)
    else:  # sparse to sparse
        self.wrapped_tensor.add_(other.wrapped_tensor.to_dense(), alpha=alpha)


@implements(torch.Tensor.mul_)
def torch_tensor_mul_(base_impl, func, types, self, other):
    if get_format(self) == torch.Tensor:  # sparse to dense
        self.mul_(other.wrapped_tensor.to_dense())
    elif get_format(other) in (torch.Tensor, None):  # dense to sparse
        self.wrapped_tensor.mul_(other)
    else:  # sparse to sparse
        raise Exception("Not implemented")


@implements(torch.Tensor.addcmul_)
def torch_tensor_mul_(base_impl, func, types, self, tensor1, tensor2, *, value=1):
    if get_format(self) == torch.Tensor:  # any to dense
        d1, d2 = tensor1, tensor2
        if hasattr(d1, "wrapped_tensor"):
            d1 = d1.wrapped_tensor.to_dense().to(device=d1.device)
        if hasattr(d2, "wrapped_tensor"):
            d2 = d2.wrapped_tensor.to_dense().to(device=d1.device)
        self.addcmul_(d1, d2, value=value)
    elif (
        get_format(tensor1) == torch.Tensor and get_format(tensor2) == torch.Tensor
    ):  # dense to sparse
        self.wrapped_tensor.addcmul_(tensor1, tensor2, value=value)
    else:  # sparse to sparse
        raise Exception("Not implemented")


@implements(torch.Tensor.addcdiv_)
def torch_tensor_mul_(base_impl, func, types, self, tensor1, tensor2, *, value=1):
    if get_format(self) == torch.Tensor:  # any to dense
        d1, d2 = tensor1, tensor2
        if hasattr(d1, "wrapped_tensor"):
            d1 = d1.wrapped_tensor.to_dense()
        if hasattr(d2, "wrapped_tensor"):
            d2 = d2.wrapped_tensor.to_dense()
        self.addcdiv_(d1, d2, value=value)
    elif (
        get_format(tensor1) == torch.Tensor and get_format(tensor2) == torch.Tensor
    ):  # dense to sparse
        self.wrapped_tensor.addcdiv_(tensor1, tensor2, value=value)
    else:  # sparse to sparse
        raise Exception("Not implemented")


@implements(torch._has_compatible_shallow_copy_type)
def torch__has_compatible_shallow_copy_type(base_impl, func, types, *args, **kwargs):
    return False


def densify_params(args, kwargs):
    def densify(arg):
        if isinstance(arg, SparseTensorWrapper):
            return arg.wrapped_tensor.to_dense()
        if isinstance(arg, list):
            return [densify(x) for x in arg]
        if isinstance(arg, dict):
            return {k: densify(v) for k, v in arg.items()}
        else:
            return arg

    dense_args = tuple(densify(a) for a in args)
    dense_kwargs = {k: densify(v) for k, v in kwargs.items()}

    return dense_args, dense_kwargs


@implements(
    torch.allclose,
    torch.Tensor.__repr__,
    torch._C._nn.flatten_dense_tensors,
    torch._C._nn.unflatten_dense_tensors,
)
def sparse_autoconvert_impl(base_impl, func, types, *args, **kwargs):
    dense_args, dense_kwargs = densify_params(args, kwargs)
    output = func(*dense_args, **dense_kwargs)
    return output


@implements(torch.zeros_like)
def sparse_torch_zeros_like(base_impl, func, types, *args, **kwargs):
    [tensor] = args
    if "memory_format" in kwargs:
        del kwargs["memory_format"]
    if "device" not in kwargs:
        kwargs["device"] = tensor.device
    return torch.zeros(tensor.shape, **kwargs)


##################


# wrapper for dense tensors: used to enable sparse gradients without sparsifying original tensor
class DenseTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data


# no-op sparsifier
class KeepAll:
    pass


class SameFormatSparsifier:
    def __init__(self, ref_sp_ten):
        self.ref_sp_ten = ref_sp_ten


def find_direct_container(module, weight_path):
    tokens = weight_path.split(".")
    for tok in tokens[:-1]:
        module = getattr(module, tok)
    weight = tokens[-1]
    return module, weight


def get_module_path(fully_qualified_tensor_name):
    return "".join(fully_qualified_tensor_name.rsplit(".", 1)[:-1])


def get_direct_name(fully_qualified_tensor_name):
    return fully_qualified_tensor_name.rsplit(".", 1)[-1]


class SparsityBuilder:
    def __init__(self, module):
        self.module = module
        self.traced_submodules = {}
        self.initial_weight_sparsifiers = {}
        self.weights = {}
        self.grad_weights = {}
        self.interms = {}
        self.grad_interms = {}

    def traced_submodule(self, submodule_path):
        submodule = self.module
        tokens = [x for x in submodule_path.split(".") if x]
        for token in tokens:
            submodule = getattr(submodule, token)
        if submodule_path not in self.traced_submodules:
            self.traced_submodules[submodule_path] = torch.fx.symbolic_trace(submodule)
        return self.traced_submodules[submodule_path]

    def set_weight(
        self,
        name,
        initial_sparsifier=KeepAll(),
        inline_sparsifier=KeepAll(),
        tmp_format=torch.Tensor,
        external_sparsifier=KeepAll(),
        out_format=DenseTensor,
    ):
        self.weights[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )
        self.initial_weight_sparsifiers[name] = initial_sparsifier

    def set_weight_grad(
        self,
        name,
        inline_sparsifier=KeepAll(),
        tmp_format=torch.Tensor,
        external_sparsifier=KeepAll(),
        out_format=DenseTensor,
    ):
        self.grad_weights[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )

    def set_interm(
        self,
        name,
        inline_sparsifier=KeepAll(),
        tmp_format=torch.Tensor,
        external_sparsifier=KeepAll(),
        out_format=DenseTensor,
    ):
        self.interms[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )

    def set_interm_grad(
        self,
        name,
        inline_sparsifier=KeepAll(),
        tmp_format=torch.Tensor,
        external_sparsifier=KeepAll(),
        out_format=DenseTensor,
    ):
        self.grad_interms[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )

    def fill_remaining(self):
        # change remaining formats of tensor/gradient pairs
        weight_names = set(self.weights.keys())
        grad_weight_names = set(self.grad_weights.keys())
        for weight_name in weight_names - grad_weight_names:
            self.set_weight_grad(
                weight_name, KeepAll(), torch.Tensor, KeepAll(), DenseTensor
            )
        for grad_weight_name in grad_weight_names - weight_names:
            self.set_weight(
                grad_weight_name,
                KeepAll(),
                KeepAll(),
                torch.Tensor,
                KeepAll(),
                DenseTensor,
            )
        interm_names = set(self.interms.keys())
        grad_interm_names = set(self.grad_interms.keys())
        for grad_interm_name in grad_interm_names - interm_names:
            self.set_interm(
                grad_interm_name, KeepAll(), torch.Tensor, KeepAll(), DenseTensor
            )
        for interm_name in interm_names - grad_interm_names:
            self.set_interm_grad(
                interm_name, KeepAll(), torch.Tensor, KeepAll(), DenseTensor
            )

    def replace_with_traced_submodules(self, sparse_module):
        for traced_module in self.traced_submodules.values():
            traced_module.recompile()

        # start replacing submodules from the outermost to innermost
        def num_tokens(x):
            path, module = x
            return len([t for t in path.split(".") if t])

        traced_submodules_list = sorted(
            list(self.traced_submodules.items()), key=num_tokens
        )

        for module_path, traced_module in traced_submodules_list:
            if module_path == "":
                sparse_module = traced_module
            else:
                module_parent, module_name = find_direct_container(
                    sparse_module, module_path
                )
                if not hasattr(module_parent, module_name):
                    raise KeyError(f"Can't find module under {module_path} path")
                setattr(module_parent, module_name, traced_module)

        return sparse_module

    def get_sparse_model(self):
        self.fill_remaining()
        with torch.no_grad():
            sparse_module = copy.deepcopy(self.module)

            for name, (sp1, fmt1, sp2, fmt2) in self.weights.items():
                grad_sp1, grad_fmt1, grad_sp2, grad_fmt2 = self.grad_weights[name]

                direct_module, direct_name = find_direct_container(sparse_module, name)
                original_tensor = getattr(direct_module, direct_name)
                initial_sparsifier = self.initial_weight_sparsifiers[name]
                initial_sparsifier_instance = get_sparsifier_implementation(
                    initial_sparsifier.__class__, torch.Tensor, fmt2
                )
                sparse_tensor = initial_sparsifier_instance(
                    initial_sparsifier,
                    original_tensor,
                    (grad_sp1, grad_fmt1, grad_sp2, grad_fmt2),
                )
                wrapper = SparseParameterWrapper(sparse_tensor)
                setattr(direct_module, direct_name, wrapper)

            for name, (sp1, fmt1, sp2, fmt2) in self.interms.items():
                grad_sp1, grad_fmt1, grad_sp2, grad_fmt2 = self.grad_interms[name]

                direct_name = get_direct_name(name)
                submodule_path = get_module_path(name)
                submodule = self.traced_submodule(submodule_path)

                [node] = (n for n in submodule.graph.nodes if n.name == direct_name)
                node.target = sparsified_op(
                    node.target,
                    [(sp1, fmt1, sp2, fmt2)],
                    [(grad_sp1, grad_fmt1, grad_sp2, grad_fmt2)],
                )

            sparse_module = self.replace_with_traced_submodules(sparse_module)

            return sparse_module


# TODO: clean the code above


def get_format(tensor):
    if hasattr(tensor, "wrapped_tensor"):
        return tensor.wrapped_tensor.__class__
    if isinstance(tensor, torch.Tensor):
        return torch.Tensor
    return None


def has_format(tensor, format):
    if not hasattr(tensor, "wrapped_tensor"):
        if isinstance(tensor, torch.Tensor) and format == torch.Tensor:
            return True
        return format is None
    if tensor.wrapped_tensor.__class__ == format:
        return True
    return False


SPARSIFIER_IMPLEMENTATIONS = {}


def register_sparsifier_implementation(sparsifer, inp, out):
    def decorator(func):
        if (sparsifer, inp, out) in SPARSIFIER_IMPLEMENTATIONS:
            raise Exception(
                "Trying to register sparsifier implementation second time! Use unregister_sparsifier_implementation to remove existing implementation."
            )
        SPARSIFIER_IMPLEMENTATIONS[(sparsifer, inp, out)] = func
        return func  # func is not modified

    return decorator


def unregister_sparsifier_implementation(sparsifer, inp, out):
    del SPARSIFIER_IMPLEMENTATIONS[(sparsifer, inp, out)]


def get_sparsifier_implementation(sparsifier, inp, out):
    # handle special cases
    not_a_tensor = (sparsifier, inp, out) == (None.__class__, None, None)
    nothing_to_do = (sparsifier == KeepAll) and (inp == out)
    if not_a_tensor or nothing_to_do:
        return lambda sp, ten: ten
    # general case
    impl = SPARSIFIER_IMPLEMENTATIONS.get((sparsifier, inp, out))
    if impl is None:
        err_msg = f"Sparsifier implementation is not registered. sparsifier: {sparsifier} inp: {inp} out: {out}."
        if DISPATCH_FAILURE == DISPATCH_WARN and inp != torch.Tensor:
            logging.warning(
                f"{err_msg} Use fallback implementation. {sparsifier} inp: {torch.Tensor} out: {out}"
            )
            # we can try to use fallback implementation input_format1 -> torch.Tensor -> sparsifier -> input_format2
            # instead of input_format1 -> sparsifier -> input_format2
            fallback_sparsifier = get_sparsifier_implementation(
                (sparsifier, torch.Tensor, out)
            )

            def full_sparsifier(sparsifier, tensor):
                dense_input = tensor.wrapped_tensor.to_dense()
                return fallback_sparsifier(sparsifier, dense_input)

            return full_sparsifier
        else:
            raise DispatchError(err_msg)
    return impl


FWD_OP_IMPLS = {}


def register_fwd_op_impl(operator, inp, out):
    def decorator(func):
        FWD_OP_IMPLS[(operator, tuple(inp), tuple(out))] = func
        return func

    return decorator


def pretty_name(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)([pretty_name(o) for o in obj])
    if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        return f"{obj.__module__}.{obj.__qualname__}"
    return str(obj)


def get_fwd_op_impl(operator, inp, out):
    impl = FWD_OP_IMPLS.get((operator, tuple(inp), tuple(out)))
    if impl is None:
        inp_str = pretty_name(inp)
        out_str = pretty_name(out)
        err_msg = f"Sparse operator implementation is not registered (fwd). op: {pretty_name(operator)} inp: {inp_str} out: {out_str}."
        raise DispatchError(err_msg)
    return impl


BWD_OP_IMPLS = {}


def register_bwd_op_impl(operator, grad_out, grad_inp, inp):
    def decorator(func):
        BWD_OP_IMPLS[(operator, tuple(grad_out), tuple(grad_inp), tuple(inp))] = func
        return func

    return decorator


def get_bwd_op_impl(operator, grad_out, grad_inp, inp):
    impl = BWD_OP_IMPLS.get((operator, tuple(grad_out), tuple(grad_inp), tuple(inp)))
    if impl is None:
        grad_out_str = pretty_name(grad_out)
        grad_inp_str = pretty_name(grad_inp)
        inp_str = pretty_name(inp)
        err_msg = f"Sparse operator implementation is not registered (bwd). op: {pretty_name(operator)} grad_out: {grad_out_str} grad_inp: {grad_inp_str} inp: {inp_str}."
        raise DispatchError(err_msg)
    return impl


def get_func_signature(original_func):
    override_dict = torch.overrides.get_testing_overrides()
    dummy_func = override_dict[original_func]
    signature = inspect.signature(dummy_func)
    return signature


def simplify_tensor_tuple(tensors):
    singular_sequence = isinstance(tensors, (tuple, list)) and len(tensors) == 1
    return tensors[0] if singular_sequence else tuple(tensors)


def canonicalize_tensor_tuple(tensors):
    singular_sequence = isinstance(tensors, (tuple, list))
    return tensors if singular_sequence else (tensors,)


def check_formats(op_instance, tensors, target_formats):
    matches = tuple(has_format(t, f) for t, f in zip(tensors, target_formats))
    returned_formats = tuple(t.__class__ for t in tensors)
    if not all(matches):
        raise KeyError(
            f"Operator implementation {op_instance} returned outputs in incorrect formats. Expected: {target_formats}. Returned: {returned_formats}."
        )


def expand_none_tuples(tuple_list, dims):
    return tuple(((None,) * dims if t is None else t) for t in tuple_list)


def collapse_none_tuples(tuple_list):
    def is_tuple_of_none(tt):
        return all((t is None) for t in tt)

    return tuple((None if is_tuple_of_none(t) else t) for t in tuple_list)


def create_fallback_fwd_impl(out_fmts):
    def fallback_fwd_impl(ctx, inputs, output_sparsifiers):
        fallback_inputs, _ = densify_params(inputs, {})
        fallback_inputs = tuple(
            (inp.detach() if isinstance(inp, torch.Tensor) else inp)
            for inp in fallback_inputs
        )
        for inp in fallback_inputs:
            if isinstance(inp, torch.Tensor):
                inp.requires_grad_()
        args_dict = get_func_signature(ctx.orig_op).bind(*fallback_inputs).arguments
        with torch.enable_grad():
            fallback_outputs = ctx.orig_op(**args_dict)
        fallback_outputs = canonicalize_tensor_tuple(fallback_outputs)
        outputs = []
        for out_fmt, out_sp, out in zip(out_fmts, output_sparsifiers, fallback_outputs):
            sp_impl = get_sparsifier_implementation(
                out_sp.__class__, torch.Tensor, out_fmt
            )
            outputs.append(sp_impl(out_sp, out))
        outputs = simplify_tensor_tuple(outputs)
        return outputs

    return fallback_fwd_impl


#


def create_fallback_bwd_impl(grad_inp_fmts):
    def fallback_bwd_impl(ctx, grad_outputs, input_sparsifiers):
        fallback_grad_outputs, _ = densify_params(grad_outputs, {})
        dense_fallback_inputs, _ = densify_params(ctx.saved_inputs, {})
        fallback_inputs = tuple(
            (i.detach().requires_grad_() if isinstance(i, torch.Tensor) else i)
            for i in dense_fallback_inputs
        )
        args_dict = get_func_signature(ctx.orig_op).bind(*fallback_inputs).arguments
        with torch.enable_grad():
            fallback_outputs = ctx.orig_op(**args_dict)
            fallback_outputs.backward(fallback_grad_outputs)
        fallback_grad_inputs_with_none = [
            getattr(inp, "grad", None) for inp in fallback_inputs
        ]
        grad_inputs = []
        for grad_inp_fmt, grad_inp_sp, grad_inp in zip(
            grad_inp_fmts, input_sparsifiers, fallback_grad_inputs_with_none
        ):
            if grad_inp is None:
                grad_inputs.append(None)
            else:
                sp_impl = get_sparsifier_implementation(
                    grad_inp_sp.__class__, torch.Tensor, grad_inp_fmt
                )
                grad_inputs.append(sp_impl(grad_inp_sp, grad_inp))
        return grad_inputs

    return fallback_bwd_impl


class SparseOperatorDispatcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_op, out_fmt, *args):
        def find_gradient_fmt(tensor):
            if isinstance(tensor, SparseTensorWrapper):
                if not hasattr(tensor, "grad_fmt"):
                    err_msg = "Format of gradient tensor is not set."
                    if DISPATCH_FAILURE == DISPATCH_WARN:
                        logging.warning(f"{err_msg} Fallback to dense implementation.")
                        tensor.grad_fmt = (
                            KeepAll(),
                            torch.Tensor,
                            KeepAll(),
                            DenseTensor,
                        )
                    else:
                        raise DispatchError(err_msg)
                return tensor.grad_fmt
            if isinstance(tensor, torch.Tensor):
                return (KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)
            return (None, None, None, None)

        ctx.inp_fmt = tuple(get_format(a) for a in args)
        ctx.grad_inp_fmt = tuple(find_gradient_fmt(a) for a in args)
        ctx.orig_op = orig_op
        ctx.saved_inputs = args
        sp1, fmt1, sp2, fmt2 = tuple(zip(*out_fmt))
        op_out_fmt = tuple((s.__class__, f) for s, f in zip(sp1, fmt1))
        try:
            op_impl = get_fwd_op_impl(orig_op, ctx.inp_fmt, op_out_fmt)
        except DispatchError as e:
            if DISPATCH_FAILURE == DISPATCH_WARN:
                logging.warning(f"{str(e)}. Fallback to dense implementation.")
                op_impl = create_fallback_fwd_impl(fmt1)
            else:
                raise e
        tmp_outputs = op_impl(ctx, args, sp1)
        tmp_outputs = canonicalize_tensor_tuple(tmp_outputs)
        check_formats(f"{orig_op} (fwd)", tmp_outputs, fmt1)
        assert len(tmp_outputs) == len(out_fmt)
        outputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, tmp_out)
            for tmp_out, f1, s2, f2 in zip(tmp_outputs, fmt1, sp2, fmt2)
        )
        outptus = simplify_tensor_tuple(outputs)
        return outptus

    @staticmethod
    def backward(ctx, *args):
        sp1, fmt1, sp2, fmt2 = tuple(zip(*ctx.grad_inp_fmt))
        grad_out_fmt = tuple(get_format(a) for a in args)
        op_inp_fmt = tuple(
            ((s.__class__, f) if s is not None else None) for s, f in zip(sp1, fmt1)
        )
        try:
            op_impl = get_bwd_op_impl(
                ctx.orig_op, grad_out_fmt, op_inp_fmt, ctx.inp_fmt
            )
        except DispatchError as e:
            if DISPATCH_FAILURE == DISPATCH_WARN:
                logging.warning(f"{str(e)}. Fallback to dense implementation.")
                op_impl = create_fallback_bwd_impl(fmt1)
            else:
                raise e
        tmp_grad_inputs = op_impl(ctx, args, sp1)
        tmp_grad_inputs = canonicalize_tensor_tuple(tmp_grad_inputs)
        check_formats(f"{ctx.orig_op} (bwd)", tmp_grad_inputs, fmt1)
        grad_inputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, inp)
            for inp, f1, s2, f2 in zip(tmp_grad_inputs, fmt1, sp2, fmt2)
        )
        return (None, None, *grad_inputs)


def sparsified_op(orig_op, out_fmt, grad_out_fmt):
    out_fmt = tuple(out_fmt)
    grad_out_fmt = tuple(grad_out_fmt)

    out_fmt = expand_none_tuples(out_fmt, 4)

    def sparse_op(*args, **kwargs):
        # here we want to linearize kwargs using the signature of original function
        func_sign = get_func_signature(orig_op)
        bound_args = func_sign.bind(*args, **kwargs)
        bound_args.apply_defaults()
        # arguments in the order of definition
        flat_args = bound_args.arguments.values()

        outputs = SparseOperatorDispatcher.apply(orig_op, out_fmt, *flat_args)
        outputs = canonicalize_tensor_tuple(outputs)
        for out, grad_fmt in zip(outputs, grad_out_fmt):
            if isinstance(out, SparseTensorWrapper):
                out.grad_fmt = grad_fmt
        outputs = simplify_tensor_tuple(outputs)
        return outputs

    return sparse_op


# ====================== Core implementation ======================


# ++++++++++++++++++++++ Built-in implementations ++++++++++++++++++++++

# ++++++++++++++++++++++ Custom formats ++++++++++++++++++++++


class CsrTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data.to_dense()


class CooTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data.to_dense()


class CscTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return torch.from_numpy(self.data.todense())


# ====================== Custom formats ======================

# ++++++++++++++++++++++ Custom sparsifiers ++++++++++++++++++++++


class RandomFractionSparsifier:
    def __init__(self, fraction):
        self.fraction = fraction


def random_mask_sparsify(tensor, frac):
    mask = np.random.choice([0, 1], size=tensor.shape, p=[frac, 1 - frac]).astype(
        np.float32
    )
    return tensor * torch.from_numpy(mask)


class ScalarFractionSparsifier:
    def __init__(self, fraction):
        self.fraction = fraction


def scalar_mask_sparsify(tensor, frac):
    flat_tensor = torch.flatten(tensor)
    sorted_idx = torch.argsort(torch.abs(flat_tensor))
    flat_output = torch.where(
        sorted_idx >= frac * len(flat_tensor),
        flat_tensor,
        torch.zeros_like(flat_tensor),
    )
    output = flat_output.reshape(tensor.shape)
    return output


# ====================== Custom sparsifiers ======================

# ++++++++++++++++++++++ Sparsifier implementations ++++++++++++++++++++++


@register_sparsifier_implementation(
    sparsifer=KeepAll, inp=torch.Tensor, out=DenseTensor
)
def torch_tensor_to_wrapped_dense(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(tensor),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(sparsifer=KeepAll, inp=torch.Tensor, out=CsrTensor)
def torch_tensor_to_csr(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CsrTensor(tensor.to_sparse_csr()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(sparsifer=KeepAll, inp=torch.Tensor, out=CooTensor)
def torch_tensor_to_coo(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CooTensor(tensor.to_sparse_coo()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifer=ScalarFractionSparsifier, inp=torch.Tensor, out=CooTensor
)
def torch_tensor_to_coo_scalar_fraction(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CooTensor(scalar_mask_sparsify(tensor, sparsifier.fraction).to_sparse_coo()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifer=RandomFractionSparsifier, inp=torch.Tensor, out=CooTensor
)
def torch_tensor_to_coo_random_fraction(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CooTensor(random_mask_sparsify(tensor, sparsifier.fraction).to_sparse_coo()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifer=RandomFractionSparsifier, inp=CooTensor, out=CsrTensor
)
def random_fraction_sparsifier_coo_csr(sparsifier, tensor, grad_fmt=None):
    dense = tensor.wrapped_tensor.to_dense()
    return torch_tensor_to_csr(
        KeepAll(), random_mask_sparsify(dense, sparsifier.fraction), grad_fmt
    )


@register_sparsifier_implementation(
    sparsifer=RandomFractionSparsifier, inp=torch.Tensor, out=CscTensor
)
def random_fraction_sparsifier_dense_csc(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CscTensor(
            scipy.sparse.csc_matrix(
                random_mask_sparsify(tensor, frac=sparsifier.fraction)
            )
        ),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifer=SameFormatSparsifier, inp=torch.Tensor, out=CscTensor
)
def same_format_sparsifier_dense_csc(sparsifier, tensor, grad_fmt=None):
    reference = sparsifier.ref_sp_ten
    shape = reference.wrapped_tensor.data.shape
    frac = 1 - reference.wrapped_tensor.data.nnz / (shape[0] * shape[1])
    return SparseTensorWrapper.wrapped_from_dense(
        CscTensor(scipy.sparse.csc_matrix(random_mask_sparsify(tensor, frac=frac))),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifer=RandomFractionSparsifier, inp=torch.Tensor, out=CsrTensor
)
def random_fraction_sparsifier_dense_csr(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CsrTensor(
            random_mask_sparsify(tensor, frac=sparsifier.fraction).to_sparse_csr()
        ),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifer=ScalarFractionSparsifier, inp=torch.Tensor, out=CsrTensor
)
def scalar_fraction_sparsifier_dense_csr(sparsifier, tensor, grad_fmt=None):
    return torch_tensor_to_csr(
        KeepAll(), random_mask_sparsify(tensor, sparsifier.fraction), grad_fmt
    )


# ====================== Sparsifier implementations ======================

# ++++++++++++++++++++++ Forward operator implementations ++++++++++++++++++++++
@register_fwd_op_impl(
    operator=torch.add,
    inp=(DenseTensor, CsrTensor, None, None),
    out=[(RandomFractionSparsifier, CsrTensor)],
)
def sparse_torch_add_fwd_impl(ctx, inputs, output_sparsifiers):
    input, other, alpha, out = inputs
    if out is not None:
        raise ValueError("In-place implementation is not supported")
    [out_sp] = output_sparsifiers
    dense_out = torch.add(
        input.wrapped_tensor.to_dense(),
        other.wrapped_tensor.to_dense(),
        alpha=alpha,
        out=out,
    )  # TODO make it really sparse
    return torch_tensor_to_csr(
        KeepAll(), random_mask_sparsify(dense_out, out_sp.fraction)
    )


@register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, CscTensor, torch.Tensor),
    out=[(KeepAll, torch.Tensor)],
)
def sparse_torch_nn_functional_linear_fwd_impl(ctx, inputs, output_sparsifiers):
    input, weight, bias = inputs
    ctx.save_for_backward(input, weight)
    output = torch.from_numpy(input.numpy() @ weight.wrapped_tensor.data.transpose())
    output += bias.unsqueeze(0).expand_as(output)
    return output


@register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, CsrTensor, torch.Tensor),
    out=[(KeepAll, torch.Tensor)],
)
def sparse_torch_nn_functional_linear_fwd_impl(ctx, inputs, output_sparsifiers):
    input, weight, bias = inputs
    ctx.save_for_backward(input, weight)
    input1 = input.view(-1, input.shape[-1])
    weight1 = weight.wrapped_tensor.data
    output = torch.sparse.mm(weight1, input1.T).T
    output = output.view([*input.shape[:-1], weight1.shape[0]])
    output += bias.unsqueeze(0).expand_as(output)
    return output


@register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(CooTensor, CsrTensor, torch.Tensor),
    out=[(KeepAll, torch.Tensor)],
)
def sparse_torch_nn_functional_linear_fwd_impl(ctx, inputs, output_sparsifiers):
    input, weight, bias = inputs
    ctx.save_for_backward(input, weight)
    input = input.wrapped_tensor.data.to_dense()  # TODO make it really sparse
    input1 = input.view(-1, input.shape[-1])
    weight1 = weight.wrapped_tensor.data
    output = torch.sparse.mm(weight1, input1.T).T
    output = output.view([*input.shape[:-1], weight1.shape[0]])
    output += bias.unsqueeze(0).expand_as(output)
    return output


@register_fwd_op_impl(
    operator=torch.nn.functional.gelu,
    inp=[torch.Tensor],
    out=[(RandomFractionSparsifier, CooTensor)],
)
def sparse_torch_nn_functional_gelu_fwd_impl(ctx, inputs, output_sparsifiers):
    [input] = inputs
    [sparsifier] = output_sparsifiers
    ctx.save_for_backward(input)
    output = torch.nn.functional.gelu(input)
    return torch_tensor_to_coo_random_fraction(sparsifier, output)


# ====================== Forward operator implementations ======================


# ++++++++++++++++++++++ Backward operator implementations ++++++++++++++++++++++
@register_bwd_op_impl(
    operator=torch.add,
    grad_out=[DenseTensor],
    grad_inp=(
        (RandomFractionSparsifier, CooTensor),
        (KeepAll, DenseTensor),
        None,
        None,
    ),
    inp=(DenseTensor, CsrTensor, None, None),
)
def sparse_torch_add_bwd_impl(ctx, output_grads, input_grad_sparsifiers):
    [out_grad] = output_grads
    out_grad = out_grad.wrapped_tensor.to_dense()
    sp0 = input_grad_sparsifiers[0]
    mask0 = np.random.choice(
        [0, 1], size=out_grad.shape, p=[sp0.fraction, 1 - sp0.fraction]
    ).astype(np.float32)
    return (
        torch_tensor_to_coo(KeepAll(), torch.from_numpy(mask0) * out_grad),
        torch_tensor_to_wrapped_dense(KeepAll(), out_grad),
        None,
        None,
    )


@register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=[torch.Tensor],
    grad_inp=(
        (KeepAll, torch.Tensor),
        (KeepAll, DenseTensor),
        (KeepAll, torch.Tensor),
    ),
    inp=(torch.Tensor, CscTensor, torch.Tensor),
)
def sparse_torch_nn_functional_linear_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input, weight = ctx.saved_tensors
    [grad_out] = grad_outputs
    grad_bias = grad_out.sum(0)
    grad_input = torch.from_numpy(grad_out.numpy() @ weight.wrapped_tensor.data)
    grad_weight = grad_out.T @ input
    wrapped_grad_weight = SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(grad_weight),
        grad_weight,
    )
    return grad_input, wrapped_grad_weight, grad_bias


@register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=[torch.Tensor],
    grad_inp=(
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
    ),
    inp=(torch.Tensor, CsrTensor, torch.Tensor),
)
def sparse_torch_nn_functional_linear_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input, weight = ctx.saved_tensors
    [grad_out] = grad_outputs
    grad_bias = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
    grad_out1 = grad_out.reshape(-1, grad_out.shape[-1])
    grad_input = torch.t(
        torch.sparse.mm(torch.t(weight.wrapped_tensor.data), torch.t(grad_out1))
    )
    input1 = input.view(-1, input.shape[-1])
    grad_weight = torch.sparse.mm(torch.t(grad_out1), input1)
    grad_input1 = grad_input.view(input.shape)
    return grad_input1, grad_weight, grad_bias


@register_bwd_op_impl(
    operator=torch.nn.functional.gelu,
    grad_out=[DenseTensor],
    grad_inp=[(KeepAll, torch.Tensor)],
    inp=[torch.Tensor],
)
def sparse_torch_nn_functional_gelu_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    [input] = ctx.saved_tensors
    [grad_output1] = grad_outputs
    grad_output = grad_output1.wrapped_tensor.data
    grad_input = grad_output * (
        0.5 * (1 + torch.erf(2 ** (-0.5) * input))
        + input * torch.exp(-(input**2) / 2) * (2 * torch.pi) ** (-0.5)
    )
    return grad_input


@register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=[torch.Tensor],
    grad_inp=(
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
    ),
    inp=(CooTensor, CsrTensor, torch.Tensor),
)
def sparse_torch_nn_functional_linear_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input, weight = ctx.saved_tensors
    [grad_out] = grad_outputs
    grad_bias = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
    grad_out1 = grad_out.view(-1, grad_out.shape[-1])
    grad_input = torch.t(
        torch.sparse.mm(torch.t(weight.wrapped_tensor.data), torch.t(grad_out1))
    )
    input1 = input.wrapped_tensor.to_dense()
    input1 = input1.view(-1, input1.shape[-1])
    grad_weight = torch.sparse.mm(torch.t(grad_out1), input1)
    grad_input = grad_input.reshape(input.wrapped_tensor.data.shape)
    return grad_input, grad_weight, grad_bias


# ====================== Backward operator implementations ======================

# ====================== Built-in implementations ======================
