import torch
import copy
import numpy as np
import torch.fx
import functools
import inspect
import scipy, scipy.sparse

# ++++++++++++++++++++++ Core implementation ++++++++++++++++++++++


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

    def __new__(cls, wrapped_tensor, require_grad=False):
        tensor = torch.Tensor._make_subclass(
            cls, torch.tensor([987654321.123456789]), require_grad
        )
        tensor.wrapped_tensor = wrapped_tensor
        tensor.update_grad_fix_hook()
        return tensor

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
            raise Exception(
                f"Function is not explicitly added into DEFAULT_FUNCTIONS: {func} {funcself}"
            )
        return HANDLED_FUNCTIONS[func](
            super().__torch_function__, func, types, *args, **kwargs
        )


class SparseParameterWrapper(SparseTensorWrapper, torch.nn.parameter.Parameter):
    def __new__(cls, wrapped_tensor, require_grad=True):
        return super().__new__(
            cls, wrapped_tensor=wrapped_tensor, require_grad=require_grad
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
)
def sparse_default_impl(base_impl, func, types, *args, **kwargs):
    return base_impl(func, types, args, kwargs)


@implements(torch.nn.functional.linear, torch.mm)
def sparse_operator_dispatch(base_impl, func, types, *args, **kwargs):
    op = sparsified_op(
        func,
        tuple([(KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)]),
        tuple([(KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)]),
    )
    return op(*args, **kwargs)


@implements(torch.Tensor.detach)
def sparse_copying_impl(base_impl, func, types, *args, **kwargs):
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


@implements(torch.Tensor.shape.__get__)
def sparse_redirection_property_impl(base_impl, func, types, *args, **kwargs):
    (original_tensor,) = args
    return getattr(original_tensor.wrapped_tensor, func.__self__.__name__)


@implements(torch.Tensor.size)
def sparse_redirection_function_impl(base_impl, func, types, *args, **kwargs):
    original_tensor = args[0]
    return getattr(original_tensor.wrapped_tensor, func.__name__)(*(args[1:]), **kwargs)


@implements(torch.Tensor.requires_grad_, torch.Tensor.requires_grad.__set__)
def torch_tensor_requires_grad_(base_impl, func, types, self, requries_grad=True):
    base_impl(func, types, (self, requries_grad))
    self.update_grad_fix_hook()


@implements(torch.Tensor.copy_)
def torch_tensor_copy_(base_impl, func, types, self, src, non_blocking=False):
    if get_format(self) == torch.Tensor:  # sparse to dense
        self.copy_(src.wrapped_tensor.to_dense())
    elif get_format(src) == torch.Tensor:  # dense to sparse
        self.wrapped_tensor = self.wrapped_tensor.__class__.from_dense(src)
    else:  # sparse to sparse
        converter = get_sparsifier_implementation(
            KeepAll, get_format(src), get_format(self)
        )
        dst = converter(KeepAll(), src)
        self.wrapped_tensor = dst.wrapped_tensor


@implements(torch.allclose, torch.Tensor.__repr__)
def sparse_autoconvert_impl(base_impl, func, types, *args, **kwargs):
    def densify(arg):
        if isinstance(arg, SparseTensorWrapper):
            return arg.wrapped_tensor.to_dense()
        else:
            return arg

    dense_args = tuple(densify(a) for a in args)
    dense_kwargs = {k: densify(v) for k, v in kwargs.items()}
    return base_impl(func, types, dense_args, dense_kwargs)


##################


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
        initial_sparsifier,
        inline_sparsifier,
        tmp_format,
        external_sparsifier,
        out_format,
    ):
        self.weights[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )
        self.initial_weight_sparsifiers[name] = initial_sparsifier

    def set_weight_grad(
        self, name, inline_sparsifier, tmp_format, external_sparsifier, out_format
    ):
        self.grad_weights[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )

    def set_interm(
        self, name, inline_sparsifier, tmp_format, external_sparsifier, out_format
    ):
        self.interms[name] = (
            inline_sparsifier,
            tmp_format,
            external_sparsifier,
            out_format,
        )

    def set_interm_grad(
        self, name, inline_sparsifier, tmp_format, external_sparsifier, out_format
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
            module_parent, module_name = find_direct_container(
                sparse_module, module_path
            )
            if not hasattr(module_parent, module_name):
                raise Exception(f"Can't find module under {module_parent} path")
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
                    initial_sparsifier, original_tensor
                )
                wrapper = SparseParameterWrapper(sparse_tensor.wrapped_tensor)
                wrapper.fmt = (sp1, fmt1, sp2, fmt2)
                wrapper.grad_fmt = (grad_sp1, grad_fmt1, grad_sp2, grad_fmt2)
                setattr(direct_module, direct_name, wrapper)

            for name, (sp1, fmt1, sp2, fmt2) in self.interms.items():
                grad_sp1, grad_fmt1, grad_sp2, grad_fmt2 = self.grad_interms[name]

                direct_name = get_direct_name(name)
                submodule_path = get_module_path(name)
                submodule = self.traced_submodule(submodule_path)

                [node] = (n for n in submodule.graph.nodes if n.name == direct_name)
                node.target = sparsified_op(
                    node.target,
                    tuple([(sp1, fmt1, sp2, fmt2)]),
                    tuple([(grad_sp1, grad_fmt1, grad_sp2, grad_fmt2)]),
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


# wrapper for dense tensors: used to enable sparse gradients without sparsifying original tensor
class DenseTensor:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_dense(tensor):
        return DenseTensor(tensor)

    def to_dense(self):
        return self.data


# no-op sparsifier
class KeepAll:
    pass


SPARSIFIER_IMPLEMENTATIONS = {}


def register_sparsifier_implementation(sparsifer, inp, out):
    def decorator(func):
        SPARSIFIER_IMPLEMENTATIONS[(sparsifer, inp, out)] = func
        return func  # func is not modified

    return decorator


def get_sparsifier_implementation(sparsifier, inp, out):
    # handle special cases
    not_a_tensor = (sparsifier, inp, out) == (None.__class__, None, None)
    nothing_to_do = (sparsifier == KeepAll) and (inp == out)
    if not_a_tensor or nothing_to_do:
        return lambda sp, ten: ten
    # general case
    impl = SPARSIFIER_IMPLEMENTATIONS.get((sparsifier, inp, out))
    if impl is None:
        raise KeyError(
            f"Sparsifier implementation is not registered. sparsifier: {sparsifier} inp: {inp} out: {out}"
        )
        assert 0
    return impl


# register trivial sparsifiers
@register_sparsifier_implementation(KeepAll, torch.Tensor, DenseTensor)
def trivial_sparsifier_impl(s, t):
    return SparseTensorWrapper(DenseTensor.from_dense(t))


FWD_OP_IMPLS = {}


def register_fwd_op_impl(operator, inp, out):
    def decorator(func):
        FWD_OP_IMPLS[(operator, inp, out)] = func
        return func

    return decorator


def get_fwd_op_impl(operator, inp, out):
    impl = FWD_OP_IMPLS.get((operator, inp, out))
    if impl is None:
        raise KeyError(
            f"Sparse operator implementation is not registered (fwd). op: {operator} inp: {inp} out: {out}"
        )
    return impl


BWD_OP_IMPLS = {}


def register_bwd_op_impl(operator, grad_out, grad_inp, inp):
    def decorator(func):
        BWD_OP_IMPLS[(operator, grad_out, grad_inp, inp)] = func
        return func

    return decorator


def get_bwd_op_impl(operator, grad_out, grad_inp, inp):
    impl = BWD_OP_IMPLS.get((operator, grad_out, grad_inp, inp))
    if impl is None:
        raise KeyError(
            f"Sparse operator implementation is not registered (bwd). op: {operator} grad_out: {grad_out} grad_inp: {grad_inp} inp: {inp}"
        )
    return impl


FUNC_WRAPPERS = {}


def register_func_wrapper(original_func):
    def impl(func):
        FUNC_WRAPPERS[original_func] = func
        return func

    return impl


def get_func_wrapper(original_func):
    f = FUNC_WRAPPERS.get(original_func)
    if f is not None:
        return f
    try:
        signature = inspect.signature(original_func)
    except:
        raise KeyError(
            f"Function wrapper for function {original_func} is not registered and signature can't be guessed automaticaly."
        )
    raise Exception("TODO")

    def f(*args):
        return original_func.bind(args)

    return f


def simplify_tensor_tuple(tensors):
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def canonicalize_tensor_tuple(tensors):
    return tensors if isinstance(tensors, tuple) else (tensors,)


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


class SparseOperatorDispatcher(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_op, out_fmt, *args):
        def find_gradient_fmt(tensor):
            if isinstance(tensor, SparseTensorWrapper):
                if not hasattr(tensor, "grad_fmt"):
                    raise ValueError("Format of gradient tensor is not set")
                return tensor.grad_fmt
            if isinstance(tensor, torch.Tensor):
                return (KeepAll(), torch.Tensor, KeepAll(), torch.Tensor)
            return (None, None, None, None)

        ctx.inp_fmt = tuple(get_format(a) for a in args)
        ctx.grad_inp_fmt = tuple(find_gradient_fmt(a) for a in args)
        ctx.orig_op = orig_op
        sp1, fmt1, sp2, fmt2 = tuple(zip(*out_fmt))
        op_out_fmt = tuple((s.__class__, f) for s, f in zip(sp1, fmt1))
        op_impl = get_fwd_op_impl(orig_op, ctx.inp_fmt, op_out_fmt)
        tmp_outputs = op_impl(ctx, args, sp1)
        tmp_outputs = canonicalize_tensor_tuple(tmp_outputs)
        check_formats(f"{orig_op} (fwd)", tmp_outputs, fmt1)
        assert len(tmp_outputs) == len(out_fmt)
        outputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, tmp_out)
            for tmp_out, f1, s2, f2 in zip(tmp_outputs, fmt1, sp2, fmt2)
        )
        return simplify_tensor_tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        sp1, fmt1, sp2, fmt2 = tuple(zip(*ctx.grad_inp_fmt))
        grad_out_fmt = tuple(get_format(a) for a in args)
        op_inp_fmt = tuple(
            ((s.__class__, f) if s is not None else None) for s, f in zip(sp1, fmt1)
        )
        op_impl = get_bwd_op_impl(ctx.orig_op, grad_out_fmt, op_inp_fmt, ctx.inp_fmt)
        tmp_grad_inputs = op_impl(ctx, args, sp1)
        tmp_grad_inputs = canonicalize_tensor_tuple(tmp_grad_inputs)
        check_formats(f"{ctx.orig_op} (bwd)", tmp_grad_inputs, fmt1)
        grad_inputs = tuple(
            get_sparsifier_implementation(s2.__class__, f1, f2)(s2, inp)
            for inp, f1, s2, f2 in zip(tmp_grad_inputs, fmt1, sp2, fmt2)
        )
        return (None, None, *grad_inputs)


def sparsified_op(orig_op, out_fmt, grad_out_fmt):
    out_fmt = expand_none_tuples(out_fmt, 4)

    def sparse_op(*args, **kwargs):
        # here we want to linearize kwargs using the signature of original function
        func_wrapper = get_func_wrapper(orig_op)
        func_sign = inspect.signature(func_wrapper)
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

    @staticmethod
    def from_dense(tensor):
        return CsrTensor(tensor.to_sparse_csr())

    def to_dense(self):
        return self.data.to_dense()


class CooTensor:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_dense(tensor):
        return CooTensor(tensor.to_sparse_coo())

    def to_dense(self):
        return self.data.to_dense()


class CscTensor:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def from_dense(tensor):
        return CscTensor(scipy.sparse.csc_matrix(tensor))

    def to_dense(self):
        return torch.from_numpy(self.data.todense())

    @property
    def shape(self):
        return torch.Size(self.data.shape)

    def size(self):
        return torch.Size(
            self.data.shape
        )  # It is not a bug: PyTorch expects shape when asked for size


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
    sparsifer=RandomFractionSparsifier, inp=CooTensor, out=CsrTensor
)
def random_fraction_sparsifier_coo_csr(sparsifier, tensor):
    return SparseTensorWrapper(
        CsrTensor.from_dense(
            random_mask_sparsify(tensor.wrapped_tensor.to_dense(), sparsifier.fraction)
        )
    )


@register_sparsifier_implementation(
    sparsifer=RandomFractionSparsifier, inp=torch.Tensor, out=CscTensor
)
def random_fraction_sparsifier_dense_csc(sparsifier, tensor):
    return SparseTensorWrapper(
        CscTensor.from_dense(random_mask_sparsify(tensor, frac=sparsifier.fraction))
    )


@register_sparsifier_implementation(
    sparsifer=RandomFractionSparsifier, inp=torch.Tensor, out=CsrTensor
)
def random_fraction_sparsifier_dense_csr(sparsifier, tensor):
    return SparseTensorWrapper(
        CsrTensor.from_dense(random_mask_sparsify(tensor, frac=sparsifier.fraction))
    )


@register_sparsifier_implementation(
    sparsifer=ScalarFractionSparsifier, inp=torch.Tensor, out=CsrTensor
)
def scalar_fraction_sparsifier_dense_csr(sparsifier, tensor):
    return SparseTensorWrapper(
        CsrTensor.from_dense(scalar_mask_sparsify(tensor, sparsifier.fraction))
    )


# ====================== Sparsifier implementations ======================

# ++++++++++++++++++++++ Operator signatures ++++++++++++++++++++++


@register_func_wrapper(torch.add)
def torch_add_sign(input, other, *, alpha=1, out=None):
    return torch.add(input, other, alpha=alpha, out=out)


@register_func_wrapper(torch.nn.functional.linear)
def torch_nn_functional_linear_sign(input, weight, bias=None):
    return torch.nn.functional.linear(input, weight, bias)


@register_func_wrapper(torch.nn.functional.gelu)
def torch_add_sign(input):
    return torch.nn.functional.gelu(input)


@register_func_wrapper(torch.mm)
def torch_nn_functional_linear_sign(input, mat2):
    return torch.mm(input, mat2)


# ====================== Operator signatures ======================

# ++++++++++++++++++++++ Forward operator implementations ++++++++++++++++++++++
@register_fwd_op_impl(
    operator=torch.add,
    inp=(DenseTensor, CsrTensor, None, None),
    out=((RandomFractionSparsifier, CsrTensor),),
)
def sparse_torch_add_fwd_impl(ctx, inputs, output_sparsifiers):
    input, other, alpha, out = inputs
    if out is not None:
        raise Exception("In-place implementation is not supported")
    (out_sp,) = output_sparsifiers
    dense_out = torch.add(
        input.wrapped_tensor.to_dense(),
        other.wrapped_tensor.to_dense(),
        alpha=alpha,
        out=out,
    )  # TODO make it really sparse
    csr_out = CsrTensor.from_dense(random_mask_sparsify(dense_out, out_sp.fraction))
    return SparseTensorWrapper(csr_out)


@register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, CscTensor, torch.Tensor),
    out=((KeepAll, torch.Tensor),),
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
    out=((KeepAll, torch.Tensor),),
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
    out=((KeepAll, torch.Tensor),),
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
    inp=tuple([torch.Tensor]),
    out=tuple([(RandomFractionSparsifier, CooTensor)]),
)
def sparse_torch_nn_functional_gelu_fwd_impl(ctx, inputs, output_sparsifiers):
    [input] = inputs
    [sparsifier] = output_sparsifiers
    ctx.save_for_backward(input)
    output = torch.nn.functional.gelu(input)
    return SparseTensorWrapper(
        CooTensor.from_dense(random_mask_sparsify(output, sparsifier.fraction))
    )


# ====================== Forward operator implementations ======================


# ++++++++++++++++++++++ Backward operator implementations ++++++++++++++++++++++
@register_bwd_op_impl(
    operator=torch.add,
    grad_out=(DenseTensor,),
    grad_inp=(
        (RandomFractionSparsifier, CooTensor),
        (KeepAll, DenseTensor),
        None,
        None,
    ),
    inp=(DenseTensor, CsrTensor, None, None),
)
def sparse_torch_add_bwd_impl(ctx, output_grads, input_grad_sparsifiers):
    (out_grad,) = output_grads
    out_grad = out_grad.wrapped_tensor.to_dense()
    sp0 = input_grad_sparsifiers[0]
    mask0 = np.random.choice(
        [0, 1], size=out_grad.shape, p=[sp0.fraction, 1 - sp0.fraction]
    ).astype(np.float32)
    in1_grad = CooTensor.from_dense(torch.from_numpy(mask0) * out_grad)
    in2_grad = DenseTensor.from_dense(out_grad)
    return (
        SparseTensorWrapper(in1_grad),
        SparseTensorWrapper(in2_grad),
        None,
        None,
    )


@register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=(torch.Tensor,),
    grad_inp=(
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
    ),
    inp=(torch.Tensor, CscTensor, torch.Tensor),
)
def sparse_torch_nn_functional_linear_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input, weight = ctx.saved_tensors
    (grad_out,) = grad_outputs
    grad_bias = grad_out.sum(0)
    grad_input = torch.from_numpy(grad_out.numpy() @ weight.wrapped_tensor.data)
    grad_weight = grad_out.T @ input
    return grad_input, grad_weight, grad_bias


@register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=tuple([torch.Tensor]),
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
    grad_out=tuple([DenseTensor]),
    grad_inp=tuple([(KeepAll, torch.Tensor)]),
    inp=tuple([torch.Tensor]),
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
    grad_out=(torch.Tensor,),
    grad_inp=(
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
        (KeepAll, torch.Tensor),
    ),
    inp=(CooTensor, CsrTensor, torch.Tensor),
)
def sparse_torch_nn_functional_linear_bwd_impl(ctx, grad_outputs, input_sparsifiers):
    input, weight = ctx.saved_tensors
    (grad_out,) = grad_outputs
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
