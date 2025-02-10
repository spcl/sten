# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/examples/rosa.py
class RoSASparsifier:
    def __init__(self, config):
        self.config = config


def grad_to_mask_fn(density, grad):
    idx = torch.topk(torch.abs(grad.flatten()).float(), int(density * grad.numel()), sorted=False).indices
    mask = torch.zeros_like(grad.flatten())
    mask.scatter_(0, idx, 1.)
    mask = mask.reshape_as(grad).bool()
    return mask


class RoSATensor:
    def __init__(self, config, shape, device):
        self.out_features, self.in_features = shape
        self.config = config
        self.orig_dense = None
        self.device = device
        self.lora_alpha = config['lora_alpha']
        self.scaling = self.lora_alpha / config['lora_r']
        self.lora_dropout = config['lora_dropout']
        self.training = True
        
        self.dtype = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32
        }[config['rosa_dtype']]
        
        # LoRA
        self.lora_r = config['lora_r']
        self.LA = None  # None means LORA is disabled and not in use
        self.LB = None

        # For Spa
        self.shape = (self.out_features, self.in_features)
        self.nnz = int(config['spa_d'] * self.out_features * self.in_features)
        self.S_val = None  # None means sparse part is disabled and not in use
        self.S_row_offs = None
        self.S_row_idx = None
        self.S_col_idx = None
        
        # schedule
        self.step = 0
        self.lora_steps = int(self.config['schedule'].split('wl')[-1])

    def reset_orig_parameter(self, orig_dense):
        self.orig_dense = orig_dense

    def reset_lora_parameters(self):
        self.LA = torch.empty((self.config['lora_r'], self.in_features), dtype=self.dtype, device=self.device)
        self.LB = torch.empty((self.out_features, self.config['lora_r']), dtype=self.dtype, device=self.device)
        torch.nn.init.normal_(self.LA, std=1 / self.lora_r)
        torch.nn.init.zeros_(self.LB)

    def reset_sparse_parameters(self, grad):
        mask = grad_to_mask_fn(self.config['spa_d'], grad)
        self.set_mask(mask)

    def set_mask(self, mask):
        nnz = mask.sum().int().item()
        assert self.nnz == nnz, f'mask.nnz does not match the numel of spa values. mask.nnz: {nnz}, self.nnz: {self.nnz}'
        assert mask.shape == self.shape, f'mask.shape does not match spa.shape. mask.shape: {mask.shape}, spa.shape: {self.shape}'
        
        sparse_tensor = csr_matrix(mask.cpu())
        self.S_val = torch.zeros((self.nnz, ), dtype=self.dtype, device=self.device)
        self.S_row_offs = torch.tensor(sparse_tensor.indptr, dtype=torch.int32, device=self.device)
        self.S_col_idx = torch.tensor(sparse_tensor.indices, dtype=torch.int16, device=self.device)
        self.S_row_idx = torch.argsort(-1 * torch.diff(self.S_row_offs)).to(torch.int16)

    def to_dense(self):
        res = self.orig_dense if self.orig_dense is not None else torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self.LA is not None:
            res += (self.LB @ self.LA) * self.scaling
        if self.S_val is not None:
            res += torch.sparse_csr_tensor(
                self.S_row_offs.to(torch.int64),
                self.S_col_idx.to(torch.int64),
                self.S_val.data,
                size=self.shape,
                dtype=self.dtype,
                device=self.device
            ).to_dense()
        return res
    
    def add_(self, other, *, alpha=1):
        if not hasattr(other, 'wrapped_tensor') or not isinstance(other.wrapped_tensor, RoSATensor):
            raise NotImplementedError("Only RoSATensor addition as a gradient is supported at the moment")
        other_rosa = other.wrapped_tensor
        if other_rosa.S_val is not None:
            if self.S_val is None:
                self.S_val = other_rosa.S_val * alpha
                self.S_row_offs = other_rosa.S_row_offs
                self.S_row_idx = other_rosa.S_row_idx
                self.S_col_idx = other_rosa.S_col_idx
            else:
                self.S_val += other_rosa.S_val * alpha
        if other_rosa.LA is not None:
            if self.LA is None:
                self.LA = other_rosa.LA * alpha
                self.LB = other_rosa.LB * alpha
            else:
                self.LA += other_rosa.LA * alpha
                self.LB += other_rosa.LB * alpha
        return self
    
    def mul_(self, other):
        if not isinstance(other, float):
            raise NotImplementedError("Only scalar multiplication is supported at the moment")
        if self.S_val is not None:
            self.S_val *= other
        if self.LA is not None:
            self.LA *= other
            self.LB *= other
        return self


@sten.register_sparsifier_implementation(
    sparsifier=RoSASparsifier, inp=torch.Tensor, out=RoSATensor
)
def rosa_sparsifier_impl(sparsifier, tensor, grad_fmt=None):
    rosa_tensor = RoSATensor(
        sparsifier.config,
        tensor.shape,
        tensor.device,
    )
    rosa_tensor.reset_orig_parameter(tensor)
    rosa_tensor.reset_lora_parameters()
    return sten.SparseTensorWrapper.wrapped_from_dense(
        rosa_tensor,
        tensor,
        grad_fmt,
    )


@sten.register_sparsifier_implementation(
    sparsifier=sten.sten.SameFormatSparsifier, inp=torch.Tensor, out=RoSATensor
)
def my_sparsifier_implementation(sparsifier, tensor, grad_fmt=None):
    raise NotImplementedError("TODO")


@sten.register_fwd_op_impl(
    operator=torch.nn.functional.linear,
    inp=(torch.Tensor, RoSATensor),  
    out=None,  # default (dense)
)
def my_operator(ctx, inputs, output_sparsifiers):
    X, W = inputs
    WW = W.wrapped_tensor
    
    input_shape = X.shape
    X = X.reshape(-1, X.shape[-1])
    
    orig_W = WW.orig_dense
    
    WW.step += 1

    needs_4bit_deq = False
    b = None
    if orig_W.dtype in [torch.bfloat16, torch.float16, torch.float32]:
        W = orig_W.to(X.dtype)
    else:
        needs_4bit_deq = True
        W = bnb.functional.dequantize_4bit(orig_W.data, orig_W.quant_state).to(X.dtype)
    
    if WW.S_val is None:
        O = torch.mm(X, W.T)
    else:
        O = torch.mm(X, csr_add(WW.S_val, WW.S_row_offs, WW.S_row_idx, WW.S_col_idx, W).T)

    if b is not None:
        O += b.to(X.dtype).unsqueeze(0)
    
    keep_prob = None
    D = None # the dropout mask
    if WW.LA is not None:
        LA = WW.LA
        LB = WW.LB
        if WW.training:
            keep_prob = 1 - WW.lora_dropout
            D = torch.rand_like(X) < keep_prob
            O += WW.lora_dropout * torch.mm(torch.mm((X * D) / keep_prob, LA.T), LB.T)
        else:
            O += WW.lora_dropout * torch.mm(torch.mm(X, LA.T), LB.T)

    ctx.save_for_backward(X, orig_W, WW.LA, WW.LB, WW.S_val, WW.S_row_offs, WW.S_row_idx, WW.S_col_idx, D)
    ctx.needs_4bit_deq = needs_4bit_deq
    ctx.input_shape = input_shape
    ctx.lora_scaling = WW.scaling
    ctx.keep_prob = keep_prob
    ctx.step = WW.step
    ctx.lora_steps = WW.lora_steps
    ctx.rosa_config = WW.config
    
    return O.reshape(*input_shape[:-1], O.shape[-1])


@sten.register_bwd_op_impl(
    operator=torch.nn.functional.linear,
    grad_out=None,  # default (dense)
    grad_inp=((sten.sten.KeepAll, torch.Tensor), (sten.sten.KeepAll, RoSATensor)),
    inp=(torch.Tensor, RoSATensor),  
)
def my_operator(ctx, grad_outputs, input_sparsifiers):
    [dO] = grad_outputs
    
    dO = dO.reshape(-1, dO.shape[-1])
    X, orig_W, LA, LB, S_val, S_row_offs, S_row_idx, S_col_idx, D = ctx.saved_tensors

    if ctx.needs_4bit_deq:
        W = bnb.functional.dequantize_4bit(orig_W.data, orig_W.quant_state).to(X.dtype)
    else:
        W = orig_W.to(X.dtype)
    
    if S_val is None:
        dS_val = None
        dX = torch.mm(dO, W)
    else:
        dS_val = sddmm(S_row_offs, S_row_idx, S_col_idx, dO.T.contiguous(), X.T.contiguous())
        dX = torch.mm(dO, csr_add(S_val, S_row_offs, S_row_idx, S_col_idx, W))

    if LA is not None:
        if D is None:
            dLA = ctx.lora_scaling * torch.mm(torch.mm(LB.T, dO.T), X)
            dLB = ctx.lora_scaling * torch.mm(dO.T, torch.mm(X, LA.T))
            dX += ctx.lora_scaling * torch.mm(torch.mm(dO, LB), LA)
        else:
            XD = X * D
            dLA = ctx.lora_scaling * torch.mm(torch.mm(LB.T, dO.T), XD) / ctx.keep_prob
            dLB = ctx.lora_scaling * torch.mm(dO.T, torch.mm(XD, LA.T)) / ctx.keep_prob
            dX += ctx.lora_scaling * torch.mm(torch.mm(dO, LB), LA) * D / ctx.keep_prob
    else:
        dLA = None
        dLB = None
    
    dX = dX.reshape(*ctx.input_shape)
    
    dW_rosa_tensor = RoSATensor(
        ctx.rosa_config,
        orig_W.shape,
        orig_W.device,
    )
    dW_rosa_tensor.LA = dLA
    dW_rosa_tensor.LB = dLB
    if ctx.step == ctx.lora_steps:
        # gradient collection step
        collected_grad = torch.mm(
            dO.T,
            X,
        )
        dW_rosa_tensor.reset_sparse_parameters(collected_grad)
        
    if dS_val is not None:
        dW_rosa_tensor.S_val = dS_val
        dW_rosa_tensor.S_row_offs = S_row_offs
        dW_rosa_tensor.S_row_idx = S_row_idx
        dW_rosa_tensor.S_col_idx = S_col_idx

    dW = sten.SparseTensorWrapper.wrapped_from_dense(
        dW_rosa_tensor,
        orig_W,
        grad_fmt=None,
    )
    
    grad_inputs = (dX, dW)
    return grad_inputs