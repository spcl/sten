import torch
import copy
import numpy as np
import torch.fx
import inspect
import scipy, scipy.sparse
import copy
import types
import random
import warnings
from enum import Enum
import functools
import textwrap


# ++++++++++++++++++++++ Core implementation ++++++++++++++++++++++

# ++++++++++++++++++++++ Dispatch defaults ++++++++++++++++++++++

DISPATCH_RAISE = "raise"
DISPATCH_WARN = "warn"

DISPATCH_FAILURE = DISPATCH_WARN


def set_dispatch_failure(action):
    # 'warn' or 'raise'
    if action not in (DISPATCH_RAISE, DISPATCH_WARN):
        raise ValueError("Unknown action")
    global DISPATCH_FAILURE
    DISPATCH_FAILURE = action


# ====================== Dispatch defaults ======================


class DispatchWarning(Warning):
    pass


class DispatchError(Warning):
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
            grad_fmt = (KeepAll(), torch.Tensor, KeepAll(), DenseTensor)
        return SparseTensorWrapper(
            wrapped_tensor_container=[wrapped],
            requires_grad=dense.requires_grad,
            grad_fmt=grad_fmt,
        )

    def __new__(
        cls,
        wrapped_tensor_container,
        requires_grad,
        grad_fmt,
        *,
        dummy_shape=None,
    ):
        assert not isinstance(wrapped_tensor_container[0], SparseTensorWrapper)

        # We randomly initialize tensor to help finding bugs with the incorrect use of tensor data.
        # We try to keep different shapes, but the same number of elements to avoid memory bugs that
        # may be even harder to catch.
        if dummy_shape is None:
            dummy_shape = [1] * random.randint(5, 10) + [2, 2, 3]
            random.shuffle(dummy_shape)
        wt = wrapped_tensor_container[0]
        dtype = wt.dtype if hasattr(wt, "dtype") else wt.to_dense().dtype
        device = wt.device if hasattr(wt, "device") else wt.to_dense().device
        dummy = torch.randint(770, 779, dummy_shape, dtype=dtype, device=device)

        tensor = torch.Tensor._make_subclass(
            cls,
            dummy,
            requires_grad,
        )
        tensor._wrapped_tensor_container = wrapped_tensor_container
        tensor.update_grad_fix_hook()
        tensor.grad_fmt = grad_fmt
        return tensor

    def __copy__(self):
        # shallow copy
        return sparse_tensor_builder(
            type(self),
            self._wrapped_tensor_container,
            self.requires_grad,
            self.grad_fmt,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = sparse_tensor_builder(
                type(self),
                copy.deepcopy(self._wrapped_tensor_container),
                copy.deepcopy(self.requires_grad),
                copy.deepcopy(self.grad_fmt),
            )
            if self.grad is not None:
                assert isinstance(self.grad, SparseTensorWrapper)
                grad_copy = copy.deepcopy(self.grad)
                result.grad = SparseTensorWrapper(
                    grad_copy.wrapped_tensor_container,
                    grad_copy.requires_grad,
                    grad_copy.grad_fmt,
                    dummy_shape=get_dummy_shape(result),
                )

            memo[id(self)] = result
            return result

    def __reduce_ex__(self, proto):
        # implement serialization for custom classes
        if isinstance(self, SparseTensorWrapper):
            args = (
                type(self),
                self._wrapped_tensor_container,
                self.requires_grad,
                self.grad_fmt,
            )
            return (sparse_tensor_builder, args)
        else:
            raise NotImplementedError("This should not happen.")

    def init_from_other(self, other):
        assert isinstance(other, SparseTensorWrapper)
        change_wrapper_metadata(self, other.device, other.dtype)
        self.wrapped_tensor = other.wrapped_tensor
        self.requires_grad = other.requires_grad
        self.grad_fmt = other.grad_fmt

    def __repr__(self):
        return f"SparseTensorWrapper:\n" f"{self.wrapped_tensor}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        try:
            if kwargs is None:
                kwargs = {}
            if func not in HANDLED_FUNCTIONS:
                return sparse_fallback(
                    super().__torch_function__, func, types, *args, **kwargs
                )
            return HANDLED_FUNCTIONS[func](
                super().__torch_function__, func, types, *args, **kwargs
            )
        except Exception as e:
            # Keep this error message to have an opportunity to catch the error
            # if exception is not reraised.
            warnings.warn(f"Exception raised during handling __torch_function__: {e}")
            # Sometimes this exception is silently ignored by PyTorch.
            raise e

    @property
    def wrapped_tensor(self):
        return self._wrapped_tensor_container[0]

    @wrapped_tensor.setter
    def wrapped_tensor(self, value):
        self._wrapped_tensor_container[0] = value

    def __mul__(self, rhs):
        return torch.mul(self, rhs)

    def __rmul__(self, lhs):
        return torch.mul(lhs, self)


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
            wrapped_tensor_container=sparse_tensor_wrapper._wrapped_tensor_container,
            requires_grad=sparse_tensor_wrapper.requires_grad,
            grad_fmt=sparse_tensor_wrapper.grad_fmt,
        )

    def __repr__(self):
        return "SparseParameterWrapper containing:\n" + super().__repr__()

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


# keep everything related to metadata here
@implements(
    torch.Tensor.register_hook,
    torch.Tensor.requires_grad.__get__,
    torch.Tensor._backward_hooks.__set__,
    torch.Tensor.grad.__set__,
    torch.Tensor.grad.__get__,
    torch.Tensor.is_leaf.__get__,
    torch.Tensor.grad_fn.__get__,
    torch.Tensor.__hash__,  # returns id(self) by default
    torch.Tensor.is_sparse.__get__,  # pretend to be dense
)
def sparse_default_impl(base_impl, func, types, *args, **kwargs):
    return base_impl(func, types, args, kwargs)


def get_dummy_shape(tensor):
    assert isinstance(tensor, SparseTensorWrapper)
    # ugly hack to use base class dummy implementation
    old_class = tensor.__class__
    tensor.__class__ = torch.Tensor
    dummy_shape = tensor.shape
    tensor.__class__ = old_class
    return dummy_shape


def change_wrapper_metadata(tensor, device, dtype):
    assert isinstance(tensor, SparseTensorWrapper)
    old_class = tensor.__class__
    tensor.__class__ = torch.Tensor
    tensor.data = tensor.data.to(device=device, dtype=dtype)
    tensor.__class__ = old_class


@implements(
    torch.Tensor.backward,
)
def sparse_backward_impl(base_impl, func, types, *args, **kwargs):
    # here we try to match dummy shape of gradient to the shape of self before continuing
    self = args[0]
    grad = args[1] if len(args) > 1 else kwargs["gradient"]
    match_grad = SparseTensorWrapper(
        grad._wrapped_tensor_container,
        grad.requires_grad,
        grad.grad_fmt,
        dummy_shape=get_dummy_shape(self),
    )
    new_args = [a for a in args]
    if len(new_args) > 1:
        new_args[1] = match_grad
    new_kwargs = {k: v for k, v in kwargs.items()}
    if "gradient" in new_kwargs:
        new_kwargs["gradient"] = match_grad
    return base_impl(func, types, new_args, new_kwargs)


# creates new tensor that shares data with existing (e.g. torch.Tensor.data.__get__, torch.Tensor.detach)
@implements(torch.Tensor.data.__get__, torch.Tensor.detach)
def make_new_tensor_with_data_sharing(base_impl, func, types, *args, **kwargs):
    [inp] = flattened_tensors(args) + flattened_tensors(kwargs)
    return copy.copy(inp).requires_grad_(False)


@implements(torch.Tensor.data.__set__)
def sparse_tensor_data_set(base_impl, func, types, *args, **kwargs):
    lhs, rhs = args
    # PyTorch semantics for x.data = y assignment is shallow copy.
    # We can support it when both tensors are sparse
    # but end up making deep copy with conversion when it is not possible.
    if isinstance(lhs, SparseTensorWrapper) and isinstance(rhs, SparseTensorWrapper):
        # shallow copy
        lhs.wrapped_tensor = rhs.wrapped_tensor
    else:
        # deep copy
        sparse_data_set = make_sparse_catcher(
            torch.Tensor.data.__set__, handle_inplace_modifications=True
        )
        sparse_data_set(lhs, rhs)


def sparse_tensor_builder(
    wrapper_type, wrapped_tensor_container, requires_grad, grad_fmt
):
    assert issubclass(wrapper_type, SparseTensorWrapper)
    result = SparseTensorWrapper(wrapped_tensor_container, requires_grad, grad_fmt)
    if wrapper_type == SparseParameterWrapper:
        result = SparseParameterWrapper(result)
    return result


@implements(torch.Tensor.__reduce_ex__)
def sparse_tensor__reduce_ex__(base_impl, func, types, self, proto):
    # implement serialization for custom classes
    raise NotImplementedError("Why it is not redirected automatically?")


# +++++++++++++++++++++++ fallback implementations +++++++++++++++++++++++


def apply_recurse(op, arg):
    if type(arg) in (list, tuple):
        return type(arg)(apply_recurse(op, x) for x in arg)
    if type(arg) == dict:
        return {k: apply_recurse(op, v) for k, v in arg.items()}
    else:
        return op(arg)


def apply_cond(cond_action, arg):
    for c, a in cond_action:
        if c(arg):
            return a(arg)
    return arg


def rand_as_args(arg):
    def cond(a):
        return apply_cond(
            [
                (
                    lambda x: isinstance(x, SparseTensorWrapper),
                    lambda x: torch.randn_like(x.wrapped_tensor.to_dense()),
                ),
                (
                    lambda x: isinstance(x, torch.Tensor),
                    lambda x: torch.randn_like(x),
                ),
            ],
            a,
        )

    return apply_recurse(cond, arg)


def clone_all(arg):
    def cond(a):
        return apply_cond(
            [
                (
                    lambda x: isinstance(x, SparseTensorWrapper),
                    lambda x: x.clone().detach(),
                )
            ],
            a,
        )

    return apply_recurse(cond, arg)


def flattened_tensors(args):
    flat_tensors = []

    def flatten_tensors(a):
        if isinstance(a, torch.Tensor):
            flat_tensors.append(a)

    apply_recurse(flatten_tensors, args)
    return flat_tensors


def torch_name(op):
    name = (
        torch.overrides.resolve_name(op)
        if hasattr(torch.overrides, "resolve_name")
        else None
    )
    if name is not None:
        return name

    # PyTorch < 1.12 or external function
    if hasattr(op, "__module__"):
        return f"{op.__module__}.{op.__name__}"
    elif hasattr(op, "__objclass__"):
        return f"{op.__objclass__}.{op.__name__}"
    return op.__name__


class Semantics(Enum):
    Function = 1
    Method = 2
    Attribute = 3


OP_INPLACE_SEMANTICS_OVERRIDES = {}


def is_inplace_op(op):
    if op in OP_INPLACE_SEMANTICS_OVERRIDES:
        return OP_INPLACE_SEMANTICS_OVERRIDES[op]

    # pessimistic autodetection
    if op.__name__ == "__get__":
        return False
    elif op.__name__ == "__set__":
        return True
    elif op.__name__.startswith("_"):
        # internal implementation, can do anything
        return True
    elif op.__name__.endswith("_"):
        # inplace by PyTorch notation
        return True
    else:
        return False


def get_op_semantics(op):
    if op.__name__ in ("__get__", "__set__"):
        return Semantics.Attribute
    elif isinstance(op, types.MethodDescriptorType):
        return Semantics.Method
    else:
        return Semantics.Function


def sparse_fallback(base_impl, func, types, *args, **kwargs):

    sem = get_op_semantics(func)
    inplace = is_inplace_op(func)

    if sem == Semantics.Attribute:
        if inplace:
            raise NotImplementedError("Attribute assignment is not supported")
        else:
            # attribute access, try to redirect to wrapped tensor (e.g. ten.shape)
            attribute_name = func.__self__.__name__
            self, *other_args = args
            if hasattr(self.wrapped_tensor, attribute_name):
                return getattr(self.wrapped_tensor, attribute_name)
            else:
                impl = make_sparse_catcher(func, handle_inplace_modifications=False)
                return impl(*args, **kwargs)
    elif sem == Semantics.Function:
        if inplace:
            impl = make_sparse_catcher(func, handle_inplace_modifications=True)
            return impl(*args, **kwargs)
        else:
            # functional operator with backprop
            op = sparsified_op(func, None, None)
            res = op(*args, **kwargs)
            return res
    elif sem == Semantics.Method:
        method_name = func.__name__
        self, *other_args = args
        if hasattr(self, "wrapped_tensor"):
            wrapper_class = type(self.wrapped_tensor)
            if hasattr(wrapper_class, method_name):
                impl = getattr(wrapper_class, method_name)
                return impl(self.wrapped_tensor, *other_args, **kwargs)
        # fallback
        if inplace:
            impl = make_sparse_catcher(func, handle_inplace_modifications=True)
            return impl(*args, **kwargs)
        else:
            # functional operator with backprop
            op = sparsified_op(func, None, None)
            res = op(*args, **kwargs)
            return res
    else:
        raise Exception("Unknown semantics")


# ======================= fallback implementations =======================


def resparsify_params(args, kwargs, changed_args, changed_kwargs):
    def resparsify(arg, changed_arg):
        if isinstance(arg, SparseTensorWrapper):
            sparsifier = get_sparsifier_implementation(
                SameFormatSparsifier, torch.Tensor, arg.wrapped_tensor.__class__
            )
            sparse_arg = sparsifier(SameFormatSparsifier(arg), changed_arg)
            arg.init_from_other(sparse_arg)
        elif isinstance(arg, (list, tuple)):
            for x, cx in zip(arg, changed_arg):
                resparsify(x, cx)
        elif isinstance(arg, dict):
            for x, cx in zip(arg.values(), changed_arg.values()):
                resparsify(x, cx)

    resparsify(args, changed_args)
    resparsify(kwargs, changed_kwargs)


@implements(torch.Tensor.requires_grad_, torch.Tensor.requires_grad.__set__)
def torch_tensor_requires_grad_(base_impl, func, types, self, requries_grad=True):
    res = base_impl(func, types, (self, requries_grad))
    self.update_grad_fix_hook()
    return res


def densify(args):
    def cond(a):
        return apply_cond(
            [
                (
                    lambda x: isinstance(x, SparseTensorWrapper),
                    lambda x: x.wrapped_tensor.to_dense(),
                )
            ],
            a,
        )

    return apply_recurse(cond, args)


def recurse_detach_and_enable_grad(args):
    def cond(a):
        return apply_cond(
            [
                (
                    lambda x: isinstance(x, torch.Tensor),
                    lambda x: x.detach().requires_grad_(True),
                )
            ],
            a,
        )

    return apply_recurse(cond, args)


##################


# wrapper for dense tensors: used to enable sparse gradients without sparsifying original tensor
class DenseTensor:
    def __init__(self, data):
        self.data = data

    def to_dense(self):
        return self.data


# no-op sparsifier
class KeepAll:
    """Default no-op sparsifier that does nothing.

    Useful when sparsification is not required.
    """

    def __eq__(self, other):
        return type(other) == type(self)


class SameFormatSparsifier:
    """Sparsifier used for in-place operators.

    Holds a copy of tensor before applying in-place operator.
    Allows to reapply sparsification to the tensor after the
    in-place modification with the dense result.
    """

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


class TracedSubmodules:
    def __init__(self):
        self.data = {}

    def access(self, module, submodule_path):
        submodule = module
        tokens = [x for x in submodule_path.split(".") if x]
        for token in tokens:
            submodule = getattr(submodule, token)
        if submodule_path not in self.data:
            self.data[submodule_path] = torch.fx.symbolic_trace(submodule)
        return self.data[submodule_path]


class SparsityBuilder:
    """A registry to declare tensors in the model for sparsification.

    After initialization, the weight, intermediate tensors, and their gradients
    need to be added in the registry. Finally, when all required tensors are
    marked, the new sparse model can be created either as a completely independent
    copy of original model, or as an in-place modified original model.
    In the former case, the original model can be reused, while in the latter
    it should be considered to be in a dirty unusable state and accessed only
    from the handle of returned sparse model.

    """

    def __init__(self):
        self.initial_weight_sparsifiers = {}
        self.weights = {}
        self.grad_weights = {}
        self.interms = {}
        self.grad_interms = {}

    def set_weight(
        self,
        name,
        initial_sparsifier=KeepAll(),
        inline_sparsifier=KeepAll(),
        tmp_format=torch.Tensor,
        external_sparsifier=KeepAll(),
        out_format=DenseTensor,
    ):
        """Marks the weight tensor for sparsification.

        Accepts sparsifier and the resulting format of weight tensor that should be stored in the model.

        Args:
            name (str): Fully qualified name of the weight tensor. Example: `attention.self.query.weight`.
            initial_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            out_format: The class of output tensor. Defaults to DenseTensor.
        """
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
        """Marks the weight gradient tensor for sparsification.

        Accepts inline sparsifier fused into the backward operator that outputs gradient tensor in the
        temporary tensor format. Then, temporary tensor is converted into output format using
        external sparsifier.

        Args:
            name (str): Fully qualified name of the weight tensor. Example: `attention.self.query.weight`.
            inline_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            tmp_format: The class of temporary tensor. Defaults to torch.Tensor.
            external_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            out_format: The class of output tensor. Defaults to DenseTensor.
        """
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
        """Marks the intermediate tensor for sparsification.

        Accepts inline sparsifier fused into the forward operator that outputs intermediate tensor in the
        temporary tensor format. Then, temporary tensor is converted into output format using
        external sparsifier.

        Args:
            name (str): The name of the intermediate tensor assigned by torch.fx. Example: `intermediate.gelu`.
            inline_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            tmp_format: The class of temporary tensor. Defaults to torch.Tensor.
            external_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            out_format: The class of output tensor. Defaults to DenseTensor.
        """
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
        """Marks the gradient of intermediate tensor for sparsification.

        Accepts inline sparsifier fused into the backward operator that outputs gradient of intermediate tensor in the
        temporary tensor format. Then, temporary tensor is converted into output format using
        external sparsifier.

        Args:
            name (str): The name of the intermediate tensor assigned by torch.fx. Example: `intermediate.gelu`.
            inline_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            tmp_format: The class of temporary tensor. Defaults to torch.Tensor.
            external_sparsifier: Instance of sparsifier class. Defaults to KeepAll().
            out_format: The class of output tensor. Defaults to DenseTensor.
        """
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

    def replace_with_traced_submodules(self, sparse_module, traced_submodules):
        for traced_module in traced_submodules.values():
            traced_module.recompile()

        # start replacing submodules from the outermost to innermost
        def num_tokens(x):
            path, module = x
            return len([t for t in path.split(".") if t])

        traced_submodules_list = sorted(list(traced_submodules.items()), key=num_tokens)

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

    def sparsify_model_inplace(self, module):
        """Makes marked tensors sparse in the existing PyTorch module.

        The original module passed into this function should not be used after the call.

        Example:
            sparsity_builder = SparsityBuilder()
            module = sparsity_builder.sparsify_model_inplace(module)

        Args:
            module (torch.nn.Module): PyTorch module to apply sparsification.

        Returns:
            torch.nn.Module: Sparse PyTorch module.
        """
        self.fill_remaining()

        traced_submodules = TracedSubmodules()

        with torch.no_grad():
            sparse_module = module

            for name, (sp1, fmt1, sp2, fmt2) in self.interms.items():
                grad_sp1, grad_fmt1, grad_sp2, grad_fmt2 = self.grad_interms[name]

                direct_name = get_direct_name(name)
                submodule_path = get_module_path(name)
                submodule = traced_submodules.access(module, submodule_path)

                [node] = (n for n in submodule.graph.nodes if n.name == direct_name)
                node.target = sparsified_op(
                    node.target,
                    [(sp1, fmt1, sp2, fmt2)],
                    [(grad_sp1, grad_fmt1, grad_sp2, grad_fmt2)],
                )

            sparse_module = self.replace_with_traced_submodules(
                sparse_module, traced_submodules.data
            )

            # weight modification should happen only after the call to replace_with_traced_submodules
            # otherwise these changes will be lost
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

            return sparse_module

    def sparsify_model_and_optimizer_inplace(self, module, optimizer):
        """Makes marked tensors sparse in the existing PyTorch module.
        In addition, it makes inplace updates in the optimizer,
        so it does not need to be re-created.

        The original module passed into this function should not be used after the call.

        Example:
            sparsity_builder = SparsityBuilder()
            module, optimizer = sparsity_builder.sparsify_model_inplace(module, optimizer)

        Args:
            module (torch.nn.Module): PyTorch module to apply sparsification.
            optimizer (torch.optim.Optimizer): Optimizer that references parameters of the module.

        Returns:
            torch.nn.Module: Sparse PyTorch module.
            torch.optim.Optimizer: Optimizer with updated references to parameters.
        """

        param_ids_before = []
        for param in module.parameters():
            param_ids_before.append(id(param))

        param_locations = {}
        for group_id, group in enumerate(optimizer.param_groups):
            for param_id, param in enumerate(group["params"]):
                param_locations[id(param)] = (group_id, param_id)

        module = self.sparsify_model_inplace(module)

        for param, old_id in zip(module.parameters(), param_ids_before):
            new_id = id(param)
            if old_id != new_id:
                group_id, param_id = param_locations[old_id]
                optimizer.param_groups[group_id]["params"][param_id] = param

        return module, optimizer

    def get_sparse_model(self, module):
        """Makes marked tensors sparse in the copy existing PyTorch module.

        The original module passed into this function is not modified and
        can be reused.

        Example:
            sparsity_builder = SparsityBuilder()
            sparse_module = sparsity_builder.get_sparse_model(dense_module)

        Args:
            module (torch.nn.Module): PyTorch module to apply sparsification.

        Returns:
            torch.nn.Module: Sparse PyTorch module.
        """
        copied_module = copy.deepcopy(module)
        return self.sparsify_model_inplace(copied_module)


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


def canonicalize_list_of_tensor_formats(ten_fmt_list):
    if ten_fmt_list is None or all((t is torch.Tensor) for t in ten_fmt_list):
        return None
    else:
        return tuple(ten_fmt_list)


def canonicalize_list_of_sparsifier_ten_fmt_pairs(sp_ten_fmt_list):
    if sp_ten_fmt_list is None or all(
        ((s is KeepAll) and (t is torch.Tensor)) for s, t in sp_ten_fmt_list
    ):
        return None
    else:
        return tuple(sp_ten_fmt_list)


def exact_list_of_tensor_formats(formats, tensors):
    if formats is None:
        formats = tuple(torch.Tensor for _ in tensors)
    return formats


def exact_list_of_sparsifiers(sparsifiers, tensors):
    if sparsifiers is None:
        sparsifiers = tuple(KeepAll() for _ in tensors)
    return sparsifiers


SPARSIFIER_IMPLEMENTATIONS = {}


def register_sparsifier_implementation(sparsifier, inp, out):
    """Decorator to register implementation for sparsifier.

    The registered function expects
    * sparsifier instance
    * input tensor instance
    * (optional) the format tuple (tmp sparsifier, tmp format, ext sparsifier, ext format) of gradient tensor (for output tensor) which should be returned by the backward operator.

    Example:
        @sten.register_sparsifier_implementation(
            sparsifier=MyRandomFractionSparsifier, inp=torch.Tensor, out=MyCscTensor
        )
        def torch_tensor_to_csc_random_fraction(sparsifier, tensor, grad_fmt=None):
            return sten.SparseTensorWrapper.wrapped_from_dense(
                MyCscTensor(scipy.sparse.csc_matrix(sten.random_mask_sparsify(tensor, sparsifier.fraction))),
                tensor,
                grad_fmt,
            )

    Args:
        sparsifier: Type of sparsifier.
        inp: Type of input tensor.
        out: Type of output tensor.
    """

    def decorator(func):
        if (sparsifier, inp, out) in SPARSIFIER_IMPLEMENTATIONS:
            raise Exception(
                "Trying to register sparsifier implementation second time! Use unregister_sparsifier_implementation to remove existing implementation."
            )
        SPARSIFIER_IMPLEMENTATIONS[(sparsifier, inp, out)] = func
        return func  # func is not modified

    return decorator


def unregister_sparsifier_implementation(sparsifier, inp, out):
    del SPARSIFIER_IMPLEMENTATIONS[(sparsifier, inp, out)]


def get_sparsifier_implementation(sparsifier, inp, out):
    # handle special cases
    not_a_tensor = (sparsifier, inp, out) == (None.__class__, None, None)
    nothing_to_do = (sparsifier == KeepAll) and (inp == out)
    if not_a_tensor or nothing_to_do:
        return lambda sp, ten: ten

    if (
        (sparsifier == KeepAll or sparsifier == SameFormatSparsifier)
        and (inp == torch.Tensor)
        and (out == DenseTensor)
    ):
        return lambda sp, ten: SparseTensorWrapper.wrapped_from_dense(
            DenseTensor(ten.clone().detach()), ten
        )
    # general case
    impl = SPARSIFIER_IMPLEMENTATIONS.get((sparsifier, inp, out))
    if impl is None:
        err_msg = textwrap.dedent(
            f"""\
            Sparsifier implementation is not registered:
            @sten.register_sparsifier_implementation(
                sparsifier={pretty_name(sparsifier)}, inp={pretty_name(inp)}, out={pretty_name(out)}
            )
            def my_sparsifier_implementation(sparsifier, tensor, grad_fmt=None):
                return sparsified_tensor_wrapper"""
        )
        if DISPATCH_FAILURE == DISPATCH_WARN and inp != torch.Tensor:
            warnings.warn(
                f"{err_msg} Use fallback implementation. {sparsifier} inp: {torch.Tensor} out: {out}",
                DispatchError,
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
    """Decorator to register forward implementation for operator.

    Operator implementation is expected to have the following signature:
        ctx: context defined by PyTorch operator extension API
        inputs: sequence of tensors in formats that match the inp argument of decorator.
        output_sparsifiers: sequence of sparsifier instances that match the out argument of decorator.

    Example:
        @sten.register_fwd_op_impl(
            operator=torch.mm,
            inp=(MyCscTensor, torch.Tensor),
            out=[(sten.KeepAll, torch.Tensor)],
        )
        def torch_mm_fwd_impl(ctx, inputs, output_sparsifiers):
            input1, input2 = inputs
            ctx.save_for_backward(input1, input2)
            output = torch.from_numpy(input1.wrapped_tensor.data @ input2.numpy())
            return output

    Args:
        operator: PyTorch operator (e.g. torch.add).
        inp: Tuple that holds formats (types) of input tensors.
        out: List (with size equal to the number of tensor outputs) of tuples (inline sparsifier, tmp format).
    """

    inp = canonicalize_list_of_tensor_formats(inp)
    out = canonicalize_list_of_sparsifier_ten_fmt_pairs(out)

    def decorator(func):
        FWD_OP_IMPLS[(operator, inp, out)] = func
        return func

    return decorator


def pretty_name(obj):
    if isinstance(obj, list):
        return "[" + ", ".join([pretty_name(o) for o in obj]) + "]"
    elif isinstance(obj, tuple):
        return "(" + ", ".join([pretty_name(o) for o in obj]) + ")"
    if inspect.isclass(obj):
        if obj.__module__ in ("builtins", "__main__"):
            return obj.__qualname__
        else:
            return obj.__module__ + "." + obj.__qualname__
    else:
        return str(obj)


def check_dense_inputs(inp):
    return (inp is None) or all(i in (None, torch.Tensor, DenseTensor) for i in inp)


def check_dense_outputs(out):
    if out is None:
        return True
    for so in out:
        if so is None:
            continue
        s, o = so
        if s != KeepAll:
            return False
        if o not in (torch.Tensor, DenseTensor):
            return False
    return True


def get_fwd_op_impl(operator, inp, out):
    inp = canonicalize_list_of_tensor_formats(inp)
    out = canonicalize_list_of_sparsifier_ten_fmt_pairs(out)

    impl = FWD_OP_IMPLS.get((operator, inp, out))

    if impl is None:
        inp_str = pretty_name(inp)
        out_str = pretty_name(out)
        err_msg = textwrap.dedent(
            f"""\
                Sparse operator implementation is not registered (fwd):
                @sten.register_fwd_op_impl(
                    operator={torch_name(operator)},
                    inp={inp_str},  {'# default (dense)' if inp is None else ''}
                    out={out_str},  {'# default (dense)' if out is None else ''}
                )
                def my_operator(ctx, inputs, output_sparsifiers):
                    return outputs"""
        )
        raise DispatchError(err_msg)
    return impl


BWD_OP_IMPLS = {}


def register_bwd_op_impl(operator, grad_out, grad_inp, inp):
    """Decorator to register backward implementation for operator.

    Operator implementation is expected to have the following signature:
        ctx: context defined by PyTorch operator extension API
        grad_outputs: sequence of tensors in formats that match the grad_out argument of decorator.
        input_sparsifiers: sequence of sparsifier instances that match the inp argument of decorator.

    Example:
        @sten.register_bwd_op_impl(
            operator=torch.mm,
            grad_out=[torch.Tensor],
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
            grad_input2 = torch.from_numpy(
                input1.wrapped_tensor.data.transpose() @ grad_output)
            return grad_input1, grad_input2

    Args:
        operator: PyTorch operator (e.g. torch.add).
        grad_out: Tuple of types of output gradients (inputs of the backward implementation)
        grad_inp: List (with size equal to the number of gradient inputs, which are outputs of backward implementation) of tuples (inline sparsifier, tmp format).
        inp: Tuple that holds formats (types) of input tensors from forward pass.
    """

    grad_out = canonicalize_list_of_tensor_formats(grad_out)
    grad_inp = canonicalize_list_of_sparsifier_ten_fmt_pairs(grad_inp)
    inp = canonicalize_list_of_tensor_formats(inp)

    def decorator(func):
        BWD_OP_IMPLS[(operator, grad_out, grad_inp, inp)] = func
        return func

    return decorator


def get_bwd_op_impl(operator, grad_out, grad_inp, inp):

    grad_out = canonicalize_list_of_tensor_formats(grad_out)
    grad_inp = canonicalize_list_of_sparsifier_ten_fmt_pairs(grad_inp)
    inp = canonicalize_list_of_tensor_formats(inp)

    impl = BWD_OP_IMPLS.get((operator, grad_out, grad_inp, inp))
    if impl is None:
        grad_out_str = pretty_name(grad_out)
        grad_inp_str = pretty_name(grad_inp)
        inp_str = pretty_name(inp)
        err_msg = textwrap.dedent(
            f"""\
                Sparse operator implementation is not registered (bwd):
                @sten.register_bwd_op_impl(
                    operator={torch_name(operator)},
                    grad_out={grad_out_str},  {'# default (dense)' if grad_out is None else ''}
                    grad_inp={grad_inp_str},  {'# default (dense)' if grad_inp is None else ''}
                    inp={inp_str},  {'# default (dense)' if inp is None else ''}
                )
                def my_operator(ctx, grad_outputs, input_sparsifiers):
                    return grad_inputs"""
        )
        raise DispatchError(err_msg)
    return impl


PATCHED_OVERRIDES = {
    torch.add: lambda input, other, *, out=None: -1,
    torch.Tensor.add: lambda self, other, *, out=None: -1,
    torch.stack: lambda tensors, dim=0, *, out=None: -1,
    torch.eq: lambda input, other, *, out=None: -1,
    torch.Tensor.eq: lambda input, other: -1,
    torch.zeros_like: lambda input, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=None: -1,
    torch.mul: lambda input, other, *, out=None: -1,
    torch.abs: lambda input, *, out=None: -1,
}


def bind_func_signature(original_func, args, kwargs):
    override_dict = torch.overrides.get_testing_overrides()
    dummy_func = PATCHED_OVERRIDES.get(original_func, override_dict[original_func])
    signature = inspect.signature(dummy_func)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args


def simplify_tensor_tuple(tensors):
    singular_sequence = isinstance(tensors, (tuple, list)) and len(tensors) == 1
    return tensors[0] if singular_sequence else tuple(tensors)


def canonicalize_tensor_tuple(tensors):
    singular_sequence = isinstance(tensors, (tuple, list))
    return tensors if singular_sequence else (tensors,)


def check_formats(op_instance, tensors, target_formats):
    target_formats = exact_list_of_tensor_formats(target_formats, tensors)

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
    def fallback_fwd_impl(ctx, flat_args_kwargs, output_sparsifiers):
        fallback_flat_args_kwargs = recurse_detach_and_enable_grad(
            densify(flat_args_kwargs)
        )

        flat_args = fallback_flat_args_kwargs[: ctx.disp_state.num_ten_args]
        flat_kwargs = fallback_flat_args_kwargs[ctx.disp_state.num_ten_args :]

        args = unflatten_list_of_tensors_in_args(ctx.disp_state.args_stubs, flat_args)
        kwargs = unflatten_list_of_tensors_in_args(
            ctx.disp_state.kwargs_stubs, flat_kwargs
        )
        bound_args = bind_func_signature(ctx.disp_state.orig_op, args, kwargs)
        with torch.enable_grad():
            fallback_outputs = ctx.disp_state.orig_op(
                *bound_args.args, **bound_args.kwargs
            )
        indexed_input_tensors = {
            i: inp
            for i, inp in enumerate(fallback_flat_args_kwargs)
            if isinstance(inp, torch.Tensor)
        }
        ctx.num_fwd_inputs = len(fallback_flat_args_kwargs)
        ctx.input_tensor_indices, input_tensors = zip(*indexed_input_tensors.items())

        outs_with_stabs, flat_outs = flatten_list_of_tensors_in_args(fallback_outputs)

        ctx.num_fwd_input_tensors = len(input_tensors)
        ctx.save_for_backward(*input_tensors, *flat_outs)

        out_fmts_local = out_fmts
        out_fmts_local = exact_list_of_tensor_formats(out_fmts_local, flat_outs)

        output_sparsifiers = exact_list_of_sparsifiers(output_sparsifiers, flat_outs)

        sparsified_flat_outs = []
        for out_fmt, out_sp, out in zip(out_fmts_local, output_sparsifiers, flat_outs):
            sp_impl = get_sparsifier_implementation(
                out_sp.__class__, torch.Tensor, out_fmt
            )
            sparsified_flat_outs.append(sp_impl(out_sp, out.detach()))

        outputs = unflatten_list_of_tensors_in_args(
            outs_with_stabs, sparsified_flat_outs
        )

        return outputs

    return fallback_fwd_impl


#


def create_fallback_bwd_impl(grad_inp_fmts):
    def fallback_bwd_impl(ctx, grad_outputs, input_sparsifiers):
        fallback_grad_outputs = densify(grad_outputs)
        fallback_inputs = ctx.saved_tensors[: ctx.num_fwd_input_tensors]
        fallback_outputs = ctx.saved_tensors[ctx.num_fwd_input_tensors :]
        torch.autograd.backward(fallback_outputs, grad_tensors=fallback_grad_outputs)
        fallback_grad_inputs_with_none = [None for _ in range(ctx.num_fwd_inputs)]
        fallback_grad_inputs = tuple(inp.grad for inp in fallback_inputs)
        for i, grad_inp in zip(ctx.input_tensor_indices, fallback_grad_inputs):
            fallback_grad_inputs_with_none[i] = grad_inp

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


def create_failing_bwd_impl(exception):
    def failing_bwd_impl(ctx, grad_outputs, input_sparsifiers):
        raise DispatchError(exception)

    return failing_bwd_impl


def find_gradient_fmt(tensor):
    if isinstance(tensor, SparseTensorWrapper):
        assert len(tensor.grad_fmt) == 4
        return tensor.grad_fmt
    if isinstance(tensor, torch.Tensor):
        return (KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)
    return (None, None, None, None)


def get_dummy_shapes(tensors):
    dummy_shapes = []
    for x in tensors:
        if isinstance(x, SparseTensorWrapper):
            dummy_shapes.append(get_dummy_shape(x))
        else:
            dummy_shapes.append(None)
    return dummy_shapes


def sparsifier_obj_to_classes(sparsifiers, formats):
    if sparsifiers is None and formats is None:
        return None
    return tuple(
        ((s.__class__, f) if s is not None else None)
        for s, f in zip(sparsifiers, formats)
    )


def transpose_sequence_or_none(args, num_results):
    if args is None:
        return [None] * num_results
    return tuple(zip(*args))


class SparseOperatorDispatcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, disp_state, *args_kwargs):
        ctx.dummy_input_shapes = get_dummy_shapes(args_kwargs)
        ctx.inp_fmt = tuple(get_format(a) for a in args_kwargs)
        ctx.grad_inp_fmt = tuple(find_gradient_fmt(a) for a in args_kwargs)
        ctx.disp_state = disp_state
        out_sp1, out_fmt1, out_sp2, out_fmt2 = transpose_sequence_or_none(
            ctx.disp_state.out_fmt, 4
        )
        op_out_fmt = sparsifier_obj_to_classes(out_sp1, out_fmt1)

        # warning: do not merge following section with try-except block for backward pass,
        # because their trivially_dense property may be different
        ctx.fwd_impl_fallback = False
        # find implementation for the forward pass
        try:
            op_impl_fwd = get_fwd_op_impl(disp_state.orig_op, ctx.inp_fmt, op_out_fmt)
        except DispatchError as e:
            # are all types are trivially reducible to dense?
            trivially_dense = check_dense_inputs(ctx.inp_fmt) and check_dense_outputs(
                op_out_fmt
            )

            if not trivially_dense:
                warnings.warn(
                    f"{e}\nFallback to dense implementation.", DispatchWarning
                )
                if DISPATCH_FAILURE == DISPATCH_RAISE:
                    raise e

            # try to create fallback implementation
            ctx.fwd_impl_fallback = True
            op_impl_fwd = create_fallback_fwd_impl(out_fmt1)

        tmp_outputs = op_impl_fwd(ctx, args_kwargs, out_sp1)

        out_with_stabs, flat_out = flatten_list_of_tensors_in_args(tmp_outputs)

        check_formats(f"{disp_state.orig_op} (fwd)", flat_out, out_fmt1)

        assert disp_state.out_fmt is None or len(flat_out) == len(disp_state.out_fmt)

        out_fmt1 = exact_list_of_tensor_formats(out_fmt1, flat_out)
        out_sp2 = exact_list_of_sparsifiers(out_sp2, flat_out)
        out_fmt2 = exact_list_of_tensor_formats(out_fmt2, flat_out)

        outputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, tmp_out)
            for tmp_out, f1, s2, f2 in zip(flat_out, out_fmt1, out_sp2, out_fmt2)
        )

        final_outputs = unflatten_list_of_tensors_in_args(out_with_stabs, outputs)

        return final_outputs

    @staticmethod
    def backward(ctx, *args):
        grad_out_fmt_actual = tuple(get_format(a) for a in args)
        gout_sp1, gout_fmt1, gout_sp2, gout_fmt2 = transpose_sequence_or_none(
            ctx.disp_state.grad_out_fmt, 4
        )

        if canonicalize_list_of_tensor_formats(
            gout_fmt2
        ) != canonicalize_list_of_tensor_formats(grad_out_fmt_actual):
            raise ValueError(
                f"Backward implementation received output gradients of incorrect format. Expected: {ctx.disp_state.grad_out_fmt}. Actual: {grad_out_fmt_actual}."
            )

        ginp_sp1, ginp_fmt1, ginp_sp2, ginp_fmt2 = zip(*ctx.grad_inp_fmt)

        op_ginp_fmt = sparsifier_obj_to_classes(ginp_sp1, ginp_fmt1)

        # find implementation for the backward pass
        try:
            op_impl_bwd = get_bwd_op_impl(
                ctx.disp_state.orig_op, gout_fmt2, op_ginp_fmt, ctx.inp_fmt
            )
        except DispatchError as e:
            # are all types are trivially reducible to dense?
            trivially_dense = (
                check_dense_inputs(gout_fmt2)
                and check_dense_outputs(op_ginp_fmt)
                and check_dense_inputs(ctx.inp_fmt)
            )

            if trivially_dense:
                op_impl_bwd = create_fallback_bwd_impl(ginp_fmt1)
            else:
                if DISPATCH_FAILURE == DISPATCH_RAISE:
                    raise e
                else:
                    if ctx.fwd_impl_fallback:
                        op_impl_bwd = create_fallback_bwd_impl(ginp_fmt1)
                    else:
                        op_impl_bwd = create_failing_bwd_impl(e)

        tmp_grad_inputs = op_impl_bwd(ctx, args, ginp_sp1)
        tmp_grad_inputs = canonicalize_tensor_tuple(tmp_grad_inputs)
        check_formats(f"{ctx.disp_state.orig_op} (bwd)", tmp_grad_inputs, ginp_fmt1)
        grad_inputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, inp)
            for inp, f1, s2, f2 in zip(tmp_grad_inputs, ginp_fmt1, ginp_sp2, ginp_fmt2)
        )
        # make shapes of grad_inputs the same as shapes of inputs to avoid complaints from autograd
        reshaped_grad_inputs = []
        for x, ds in zip(grad_inputs, ctx.dummy_input_shapes):
            if isinstance(x, SparseTensorWrapper):
                rx = SparseTensorWrapper(
                    wrapped_tensor_container=x._wrapped_tensor_container,
                    requires_grad=x.requires_grad,
                    grad_fmt=x.grad_fmt,
                    dummy_shape=ds,
                )
                reshaped_grad_inputs.append(rx)
            else:
                reshaped_grad_inputs.append(x)

        return (None, *reshaped_grad_inputs)


class TensorIdx:
    def __init__(self, idx):
        self.idx = idx


def flatten_list_of_tensors_in_args(args):
    flat_tensors = []

    def process_tensor(x):
        idx = len(flat_tensors)
        flat_tensors.append(x)
        return TensorIdx(idx)

    def cond(a):
        return apply_cond(
            [
                (
                    lambda x: isinstance(x, torch.Tensor),
                    process_tensor,
                ),
            ],
            a,
        )

    args_with_stubs = apply_recurse(cond, args)

    return args_with_stubs, flat_tensors


def unflatten_list_of_tensors_in_args(args_with_stubs, flat_tensors):
    def cond(a):
        return apply_cond(
            [
                (
                    lambda x: isinstance(x, TensorIdx),
                    lambda x: flat_tensors[x.idx],
                ),
            ],
            a,
        )

    args = apply_recurse(cond, args_with_stubs)
    return args


class DispatcherState:
    def __init__(
        self,
        orig_op,
        out_fmt,
        grad_out_fmt,
        args_stubs,
        kwargs_stubs,
        num_ten_args,
        num_ten_kwargs,
    ):
        self.orig_op = orig_op
        self.out_fmt = out_fmt
        self.grad_out_fmt = grad_out_fmt
        self.args_stubs = args_stubs
        self.kwargs_stubs = kwargs_stubs
        self.num_ten_args = num_ten_args
        self.num_ten_kwargs = num_ten_kwargs


class SparseOp:
    def __init__(self, orig_op, out_fmt, grad_out_fmt):
        self.orig_op = orig_op
        self.out_fmt = out_fmt
        self.grad_out_fmt = grad_out_fmt

    def __call__(self, *args, **kwargs):
        # here we want to linearize kwargs using the signature of original function
        bound_args = bind_func_signature(self.orig_op, args, kwargs)
        # arguments in the order of definition
        args_stubs, flat_args = flatten_list_of_tensors_in_args(bound_args.args)
        kwargs_stubs, flat_kwargs = flatten_list_of_tensors_in_args(bound_args.kwargs)
        disp_state = DispatcherState(
            self.orig_op,
            self.out_fmt,
            self.grad_out_fmt,
            args_stubs,
            kwargs_stubs,
            len(flat_args),
            len(flat_kwargs),
        )
        outputs = SparseOperatorDispatcher.apply(disp_state, *flat_args, *flat_kwargs)
        outputs = canonicalize_tensor_tuple(outputs)

        grad_out_fmt = self.grad_out_fmt
        grad_out_fmt = exact_list_of_tensor_formats(grad_out_fmt, outputs)

        for out, grad_fmt in zip(outputs, grad_out_fmt):
            if isinstance(out, SparseTensorWrapper):
                out.grad_fmt = grad_fmt
        outputs = simplify_tensor_tuple(outputs)
        return outputs


def sparsified_op(orig_op, out_fmt, grad_out_fmt):
    """Creates a new operator fused with sparsification and capable of returning tensors in custom formats.

    Example:
        sparse_add = sten.sparsified_op(
            orig_op=torch.add,
            out_fmt=(
                (sten.KeepAll(), torch.Tensor,
                MyRandomFractionSparsifier(0.5), MyCscTensor),
            ),
            grad_out_fmt=(
                (sten.KeepAll(), torch.Tensor,
                MyRandomFractionSparsifier(0.5), MyCscTensor),
            ),
        )

    Args:
        orig_op: Any functional operator defined in PyTorch (e.g. torch.add).
        out_fmt: Tuple of (temporary sparsifier instance, temporary tensor format class, external sparsifier instance, external tensor format class)
        grad_out_fmt: (temporary sparsifier instance, temporary tensor format class, external sparsifier instance, external tensor format class)

    Returns:
        Sparse operator with specified output tensor format.
    """

    if (out_fmt is not None) or (grad_out_fmt is not None):
        # fill default formats
        if out_fmt is None:
            out_fmt = [
                [KeepAll(), torch.Tensor, KeepAll(), torch.Tensor] for _ in grad_out_fmt
            ]
        else:
            out_fmt = [list(x) for x in out_fmt]
        if grad_out_fmt is None:
            grad_out_fmt = [
                [KeepAll(), torch.Tensor, KeepAll(), torch.Tensor] for _ in out_fmt
            ]
        else:
            grad_out_fmt = [list(x) for x in grad_out_fmt]
        # check that sparse tensors always have corresponding dense tensors in the backward pass
        for out, grad_out in zip(out_fmt, grad_out_fmt):
            if (out[3] == torch.Tensor) != (grad_out[3] == torch.Tensor):
                # automatically require DenseTensor as needed
                if out[3] == torch.Tensor:
                    out[3] = DenseTensor
                if grad_out[3] == torch.Tensor:
                    grad_out[3] = DenseTensor

    def wrapper(*args, **kwargs):
        op = SparseOp(orig_op, out_fmt, grad_out_fmt)
        return op(*args, **kwargs)

    return wrapper


@functools.cache
def make_sparse_catcher(orig_fn, handle_inplace_modifications=True):
    def sparse_catcher(*args, **kwargs):
        args_with_stubs, flat_args = flatten_list_of_tensors_in_args(args)
        kwargs_with_stubs, flat_kwargs = flatten_list_of_tensors_in_args(kwargs)
        all_flat_args = flat_args + flat_kwargs
        if any(isinstance(t, SparseTensorWrapper) for t in all_flat_args):
            # implementation that will handle args properly
            warnings.warn(
                f"Catching {torch_name(orig_fn)} called with the sparse arguments!",
                DispatchWarning,
            )
            flat_d_args = densify(flat_args)
            flat_d_kwargs = densify(flat_kwargs)
            all_flat_d_args = flat_d_args + flat_d_kwargs
            d_args = unflatten_list_of_tensors_in_args(args_with_stubs, flat_d_args)
            d_kwargs = unflatten_list_of_tensors_in_args(
                kwargs_with_stubs, flat_d_kwargs
            )
            if handle_inplace_modifications:
                arg_copies = [
                    (
                        copy.deepcopy(dense_ten)
                        if isinstance(orig_ten, SparseTensorWrapper)
                        else None
                    )
                    for orig_ten, dense_ten in zip(all_flat_args, all_flat_d_args)
                ]
            d_output = orig_fn(*d_args, **d_kwargs)
            out_with_stabs, flat_out = flatten_list_of_tensors_in_args(d_output)
            if handle_inplace_modifications:
                # check for modifications
                for cpy, orig, dense in zip(arg_copies, all_flat_args, all_flat_d_args):
                    if isinstance(orig, SparseTensorWrapper):
                        # detect changes in device or dtype first, otherwise torch.equal will raise an exception
                        # this may be required for x.data = y assignment when device or type of x can change
                        if (
                            cpy.dtype == dense.dtype
                            and cpy.device == dense.device
                            and torch.equal(cpy, dense)
                        ):
                            continue  # no inplace changes
                        sparsifier = get_sparsifier_implementation(
                            SameFormatSparsifier,
                            torch.Tensor,
                            orig.wrapped_tensor.__class__,
                        )
                        # TODO: not sure how to distinguish full replacement and nonzero modification
                        sparse_arg = sparsifier(SameFormatSparsifier(orig), dense)
                        orig.init_from_other(sparse_arg)
                    else:
                        assert cpy is None
            # return output
            if flat_out:
                flat_s_out = []
                for out in flat_out:
                    reused = None
                    for orig_inp, dense_inp in zip(all_flat_args, all_flat_d_args):
                        if out is dense_inp:
                            reused = orig_inp
                    flat_s_out.append(out if reused is None else reused)
                return unflatten_list_of_tensors_in_args(out_with_stabs, flat_s_out)
            else:
                return d_output
        else:
            # default implementation
            return orig_fn(*args, **kwargs)

    return sparse_catcher


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


class MaskedSparseTensor:
    def __init__(self, data, inplace_sparsifier):
        self.data = data
        self.sparsifier = inplace_sparsifier

    def to_dense(self):
        return self.data


# ====================== Custom formats ======================

# ++++++++++++++++++++++ Custom sparsifiers ++++++++++++++++++++++


class RandomFractionSparsifier:
    def __init__(self, fraction):
        self.fraction = fraction

    def __eq__(self, other):
        return type(self) == type(other) and self.fraction == other.fraction


def random_mask_sparsify(tensor, frac):
    mask = np.random.choice([0, 1], size=tensor.shape, p=[frac, 1 - frac]).astype(
        np.float32
    )
    return tensor * torch.from_numpy(mask).to(device=tensor.device)


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
    sparsifier=KeepAll, inp=torch.Tensor, out=DenseTensor
)
def torch_tensor_to_wrapped_dense(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        DenseTensor(tensor),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(sparsifier=KeepAll, inp=torch.Tensor, out=CsrTensor)
def torch_tensor_to_csr(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CsrTensor(tensor.to_sparse_csr()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(sparsifier=KeepAll, inp=torch.Tensor, out=CooTensor)
def torch_tensor_to_coo(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CooTensor(tensor.to_sparse_coo()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifier=ScalarFractionSparsifier, inp=torch.Tensor, out=CooTensor
)
def torch_tensor_to_coo_scalar_fraction(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CooTensor(scalar_mask_sparsify(tensor, sparsifier.fraction).to_sparse_coo()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifier=RandomFractionSparsifier, inp=torch.Tensor, out=CooTensor
)
def torch_tensor_to_coo_random_fraction(sparsifier, tensor, grad_fmt=None):
    return SparseTensorWrapper.wrapped_from_dense(
        CooTensor(random_mask_sparsify(tensor, sparsifier.fraction).to_sparse_coo()),
        tensor,
        grad_fmt,
    )


@register_sparsifier_implementation(
    sparsifier=RandomFractionSparsifier, inp=CooTensor, out=CsrTensor
)
def random_fraction_sparsifier_coo_csr(sparsifier, tensor, grad_fmt=None):
    dense = tensor.wrapped_tensor.to_dense()
    return torch_tensor_to_csr(
        KeepAll(), random_mask_sparsify(dense, sparsifier.fraction), grad_fmt
    )


@register_sparsifier_implementation(
    sparsifier=RandomFractionSparsifier, inp=torch.Tensor, out=CscTensor
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
    sparsifier=SameFormatSparsifier, inp=torch.Tensor, out=CscTensor
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
    sparsifier=RandomFractionSparsifier, inp=torch.Tensor, out=CsrTensor
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
    sparsifier=ScalarFractionSparsifier, inp=torch.Tensor, out=CsrTensor
)
def scalar_fraction_sparsifier_dense_csr(sparsifier, tensor, grad_fmt=None):
    return torch_tensor_to_csr(
        KeepAll(), random_mask_sparsify(tensor, sparsifier.fraction), grad_fmt
    )


# ====================== Sparsifier implementations ======================

# ++++++++++++++++++++++ Forward operator implementations ++++++++++++++++++++++
@register_fwd_op_impl(
    operator=torch.add,
    inp=(DenseTensor, CsrTensor),
    out=[(RandomFractionSparsifier, CsrTensor)],
)
def sparse_torch_add_fwd_impl(ctx, inputs, output_sparsifiers):
    input, other = inputs
    [out_sp] = output_sparsifiers
    dense_out = torch.add(
        input.wrapped_tensor.to_dense(),
        other.wrapped_tensor.to_dense(),
    )
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
    ),
    inp=(DenseTensor, CsrTensor),
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
    )


@register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=[torch.Tensor],
    grad_inp=(
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
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
    return grad_input, grad_weight, grad_bias


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
