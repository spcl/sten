from operator import index
import torch
import copy
import numpy as np
import torch.fx
import inspect
import scipy, scipy.sparse
import logging
import copy
import types


_log = logging.getLogger(__name__)


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
            grad_fmt = (KeepAll(), torch.Tensor, KeepAll(), DenseTensor)
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
            torch.tensor([987654321], dtype=dtype, device=device),
            requires_grad,
        )
        tensor.wrapped_tensor = wrapped_tensor
        tensor.update_grad_fix_hook()
        tensor.grad_fmt = grad_fmt
        return tensor

    def __copy__(self):
        # shallow copy
        return sparse_tensor_builder(
            type(self),
            self.wrapped_tensor,
            self.requires_grad,
            self.grad_fmt,
            self.dtype,
            self.device,
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = sparse_tensor_builder(
                type(self),
                copy.deepcopy(self.wrapped_tensor),
                copy.deepcopy(self.requires_grad),
                copy.deepcopy(self.grad_fmt),
                copy.deepcopy(self.dtype),
                copy.deepcopy(self.device),
            )
            memo[id(self)] = result
            return result

    def __reduce_ex__(self, proto):
        # implement serialization for custom classes
        if isinstance(self, SparseTensorWrapper):
            args = (
                type(self),
                self.wrapped_tensor,
                self.requires_grad,
                self.grad_fmt,
                self.dtype,
                self.device,
            )
            return (sparse_tensor_builder, args)
        else:
            raise NotImplementedError("This should not happen.")

    def init_from_other(self, other):
        assert isinstance(other, SparseTensorWrapper)
        self.wrapped_tensor = other.wrapped_tensor
        self.requires_grad = other.requires_grad
        self.grad_fmt = other.grad_fmt
        if self.dtype != other.dtype or self.device != other.device:
            raise Exception("This should never happen.")

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
            _log.error(f"Exception raised during handling __torch_function__: {e}")
            # Sometimes this exception is silently ignored by PyTorch.
            raise e


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
    torch.Tensor.backward,
    torch.Tensor.dtype.__get__,
    torch.Tensor.device.__get__,
    torch.Tensor.__hash__,  # returns id(self) by default
)
def sparse_default_impl(base_impl, func, types, *args, **kwargs):
    return base_impl(func, types, args, kwargs)


def sparse_redirect_impl(base_impl, func, _types, *args, **kwargs):
    (
        same_tensor,
        same_data,
        inplace_changes,
        num_inputs,
        num_outputs,
    ) = check_op_semantics(func, args, kwargs)

    if num_inputs == 1:
        [self] = [
            x
            for x in list(args) + list(kwargs.values())
            if isinstance(x, SparseTensorWrapper)
        ]
        if func.__name__ == "__get__":
            # attribute access, try to redirect to wrapped tensor (e.g. ten.shape)
            attribute_name = func.__self__.__name__
            if hasattr(self.wrapped_tensor, attribute_name):
                return getattr(self.wrapped_tensor, attribute_name)
        elif isinstance(func, types.MethodDescriptorType):
            # call method of a class (e.g. ten.size())
            method_name = func.__name__
            wrapper_class = type(self.wrapped_tensor)
            if hasattr(wrapper_class, method_name):
                return getattr(wrapper_class, method_name)(*args, **kwargs)

    _log.warning(
        f"Using fallback dense implementation for read-only access without tensor output: {torch.overrides.resolve_name(func)}"
    )
    d_args = densify(args)
    d_kwargs = densify(kwargs)
    return func(*d_args, **d_kwargs)


# creates new tensor that shares data with existing (e.g. torch.Tensor.data.__get__, torch.Tensor.detach)
@implements(torch.Tensor.data.__get__, torch.Tensor.detach)
def make_new_tensor_with_data_sharing(base_impl, func, types, *args, **kwargs):
    [inp] = flattened_tensors(args) + flattened_tensors(kwargs)
    return copy.copy(inp).requires_grad_(False)


def sparse_tensor_builder(
    wrapper_type, wrapped_tensor, requires_grad, grad_fmt, dtype, device
):
    assert issubclass(wrapper_type, SparseTensorWrapper)
    result = SparseTensorWrapper(wrapped_tensor, requires_grad, grad_fmt, dtype, device)
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


OP_SEMANTICS_CACHE = {}


def check_op_semantics(op, args, kwargs):
    if op in OP_SEMANTICS_CACHE:
        return OP_SEMANTICS_CACHE[op]

    _log.warning(
        f"Semantics of {torch.overrides.resolve_name(op)} is unknown, trying to discover it by executing..."
    )

    dargs = rand_as_args(args)
    dkwargs = rand_as_args(kwargs)

    flat_inputs = flattened_tensors(dargs) + flattened_tensors(dkwargs)

    copied_inputs = [t.clone().detach() for t in flat_inputs]

    output = op(*dargs, **dkwargs)

    flat_outputs = flattened_tensors(output)

    same_tensor = {}
    same_data = {}
    for inp_idx, inp in enumerate(flat_inputs):
        for out_idx, out in enumerate(flat_outputs):
            if id(inp) == id(out):
                same_tensor[inp_idx] = out_idx
            if inp.data_ptr() == out.data_ptr():
                same_data[inp_idx] = out_idx

    inplace_changes = []
    for idx, (t, r) in enumerate(zip(flat_inputs, copied_inputs)):
        if torch.any(t != r):
            inplace_changes.append(idx)

    num_inputs = len(flat_inputs)
    num_outputs = len(flat_outputs)

    result = (
        same_tensor,
        same_data,
        inplace_changes,
        num_inputs,
        num_outputs,
    )
    OP_SEMANTICS_CACHE[op] = result

    return result


def sparse_fallback(base_impl, func, types, *args, **kwargs):

    (
        same_tensor,
        same_data,
        inplace_changes,
        num_inputs,
        num_outputs,
    ) = check_op_semantics(func, args, kwargs)

    if (same_tensor == {0: 0}) and (inplace_changes == [0]) and (num_outputs == 1):
        _log.warning(
            f"Using fallback dense implementation for inplace operation: {torch.overrides.resolve_name(func)}"
        )
        # inplace operator that returns self (e.g. torch.Tensor.add_, torch.Tensor.copy_)
        d_args = densify(args)
        d_kwargs = densify(kwargs)
        d_output = func(*d_args, **d_kwargs)
        resparsify_params(args, kwargs, d_args, d_kwargs)
        output = flattened_tensors(args) + flattened_tensors(kwargs)
        return output[0]
    elif (not inplace_changes) and (num_outputs == 0) and (func.__name__ != "__set__"):
        # access some attributes of tensor (e.g. torch.Tensor.size)
        return sparse_redirect_impl(base_impl, func, types, *args, **kwargs)
    elif (not same_data) and (not inplace_changes) and (num_outputs > 0):
        # functional operator with backprop
        op = sparsified_op(
            func,
            [
                (KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)
                for _ in range(num_outputs)
            ],
            [
                (KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)
                for _ in range(num_outputs)
            ],
        )
        res = op(*args, **kwargs)
        return res
    elif (
        (not same_tensor)
        and (same_data == {0: 0})
        and (not inplace_changes)
        and (num_outputs == 1)
    ):
        # creates new tensor that shares data with existing (e.g. torch.data.__get__, torch.Tensor.detach)
        raise NotImplementedError(
            f"Probably {torch.overrides.resolve_name(func)} function should be dispatched in {make_new_tensor_with_data_sharing.__name__}. "
            "It is not added explicitly to prevent potential bugs."
        )
    else:
        raise NotImplementedError(
            f"Can't create fallback implementation for {torch.overrides.resolve_name(func)}"
        )


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
    def __eq__(self, other):
        return type(other) == type(self)


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
        err_msg = f"Sparsifier implementation is not registered. sparsifier: {sparsifier} inp: {inp} out: {out}."
        if DISPATCH_FAILURE == DISPATCH_WARN and inp != torch.Tensor:
            _log.warning(
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
    return str(obj)


def check_dense_inputs(inp):
    return all(i in (None, torch.Tensor, DenseTensor) for i in inp)


def check_dense_outputs(out):
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
    impl = FWD_OP_IMPLS.get((operator, tuple(inp), tuple(out)))
    if impl is None:
        inp_str = pretty_name(inp)
        out_str = pretty_name(out)
        err_msg = (
            f"Sparse operator implementation is not registered (fwd).\n"
            f"op: {torch.overrides.resolve_name(operator)}\n"
            f"inp: {inp_str}\n"
            f"out: {out_str}."
        )
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
        err_msg = (
            f"Sparse operator implementation is not registered (bwd).\n"
            f"    op: {pretty_name(operator)}\n"
            f"    grad_out: {grad_out_str}\n"
            f"    grad_inp: {grad_inp_str}\n"
            f"    inp: {inp_str}."
        )
        raise DispatchError(err_msg)
    return impl


PATCHED_OVERRIDES = {
    torch.add: lambda input, other, *, out=None: -1,
    torch.Tensor.add: lambda self, other, *, out=None: -1,
    torch.stack: lambda tensors, dim=0, *, out=None: -1,
    torch.eq: lambda input, other, *, out=None: -1,
    torch.Tensor.eq: lambda input, other: -1,
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
        fallback_outputs = canonicalize_tensor_tuple(fallback_outputs)
        ctx.num_fwd_input_tensors = len(input_tensors)
        ctx.save_for_backward(*input_tensors, *fallback_outputs)
        outputs = []
        for out_fmt, out_sp, out in zip(out_fmts, output_sparsifiers, fallback_outputs):
            sp_impl = get_sparsifier_implementation(
                out_sp.__class__, torch.Tensor, out_fmt
            )
            outputs.append(sp_impl(out_sp, out.detach()))
        outputs = simplify_tensor_tuple(outputs)
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
        raise exception

    return failing_bwd_impl


class SparseOperatorDispatcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, disp_state, *args_kwargs):
        def find_gradient_fmt(tensor):
            if isinstance(tensor, SparseTensorWrapper):
                assert len(tensor.grad_fmt) == 4
                return tensor.grad_fmt
            if isinstance(tensor, torch.Tensor):
                return (KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)
            return (None, None, None, None)

        inp_fmt = tuple(get_format(a) for a in args_kwargs)
        ctx.grad_inp_fmt = tuple(find_gradient_fmt(a) for a in args_kwargs)
        ctx.disp_state = disp_state
        sp1, fmt1, sp2, fmt2 = tuple(zip(*disp_state.out_fmt))
        gsp1, gfmt1, gsp2, gfmt2 = tuple(zip(*disp_state.grad_out_fmt))
        grad_inp_sp1, grad_inp_fmt1, grad_inp_sp2, grad_inp_fmt2 = tuple(
            zip(*ctx.grad_inp_fmt)
        )
        op_out_fmt = tuple(
            ((s.__class__, f) if s is not None else None) for s, f in zip(sp1, fmt1)
        )
        op_gout_fmt = tuple(
            ((s.__class__, f) if s is not None else None) for s, f in zip(gsp2, gfmt2)
        )
        op_ginp_fmt = tuple(
            ((s.__class__, f) if s is not None else None)
            for s, f in zip(grad_inp_sp1, grad_inp_fmt1)
        )

        # warning: do not merge following section with try-except block for backward pass,
        # because their trivially_dense property may be different

        fwd_impl_fallback = False

        # find implementation for the forward pass
        try:
            op_impl_fwd = get_fwd_op_impl(disp_state.orig_op, inp_fmt, op_out_fmt)
        except DispatchError as e:
            # are all types are trivially reducible to dense?
            trivially_dense = check_dense_inputs(inp_fmt) and check_dense_outputs(
                op_out_fmt
            )

            if not trivially_dense:
                _log.warning(f"{e}\nFallback to dense implementation.")
                if DISPATCH_FAILURE == DISPATCH_RAISE:
                    raise e

            # try to create fallback implementation

            fwd_impl_fallback = True
            op_impl_fwd = create_fallback_fwd_impl(fmt1)

        # find implementation for the backward pass
        try:
            ctx.op_impl_bwd = get_bwd_op_impl(
                disp_state.orig_op, gfmt2, op_ginp_fmt, inp_fmt
            )
        except DispatchError as e:
            # are all types are trivially reducible to dense?
            trivially_dense = (
                check_dense_inputs(gfmt2)
                and check_dense_outputs(op_ginp_fmt)
                and check_dense_inputs(inp_fmt)
            )

            if trivially_dense:
                ctx.op_impl_bwd = create_fallback_bwd_impl(grad_inp_fmt1)
            else:
                if DISPATCH_FAILURE == DISPATCH_RAISE:
                    raise e
                else:
                    if fwd_impl_fallback:
                        _log.warning(f"{e}\nFallback to dense implementation.")
                        ctx.op_impl_bwd = create_fallback_bwd_impl(grad_inp_fmt1)
                    else:
                        _log.warning(
                            f"{e}\nCan't fallback to dense backward implementation because custom forward implementation was used. Attempt to compute backward will raise an exception."
                        )
                        ctx.op_impl_bwd = create_failing_bwd_impl(e)

        tmp_outputs = op_impl_fwd(ctx, args_kwargs, sp1)
        tmp_outputs = canonicalize_tensor_tuple(tmp_outputs)
        check_formats(f"{disp_state.orig_op} (fwd)", tmp_outputs, fmt1)
        assert len(tmp_outputs) == len(disp_state.out_fmt)
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
        tmp_grad_inputs = ctx.op_impl_bwd(ctx, args, sp1)
        tmp_grad_inputs = canonicalize_tensor_tuple(tmp_grad_inputs)
        check_formats(f"{ctx.disp_state.orig_op} (bwd)", tmp_grad_inputs, fmt1)
        grad_inputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, inp)
            for inp, f1, s2, f2 in zip(tmp_grad_inputs, fmt1, sp2, fmt2)
        )
        return (None, *grad_inputs)


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
        for out, grad_fmt in zip(outputs, self.grad_out_fmt):
            if isinstance(out, SparseTensorWrapper):
                out.grad_fmt = grad_fmt
        outputs = simplify_tensor_tuple(outputs)
        return outputs


def sparsified_op(orig_op, out_fmt, grad_out_fmt):
    out_fmt = tuple(out_fmt)
    grad_out_fmt = tuple(grad_out_fmt)

    out_fmt = expand_none_tuples(out_fmt, 4)

    def wrapper(*args, **kwargs):
        op = SparseOp(orig_op, out_fmt, grad_out_fmt)
        return op(*args, **kwargs)

    return wrapper


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

    def __eq__(self, other):
        return type(self) == type(other) and self.fraction == other.fraction


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
        + input * torch.exp(-(input ** 2) / 2) * (2 * torch.pi) ** (-0.5)
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
