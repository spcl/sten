# https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/hooks.py
class GradCollectorHook:
    def __init__(self, name: str, module: RosaLayer, grad_acc_mode: str) -> None:
        assert grad_acc_mode in ['mean', 'mean_squared']
        self._name = name
        self._module = module
        self._grad_acc_mode = grad_acc_mode

    def __call__(self, param):
        print('hook called for', self._name)

        if not hasattr(self._module, 'collected_grad'):
            self._module.register_buffer('collected_grad', torch.zeros_like(param.grad, device='cpu'))
            setattr(self._module, 'collected_grad_cnt', 0)

        with torch.no_grad():
            prev_cnt = getattr(self._module, 'collected_grad_cnt')
            new_cnt = prev_cnt + 1

            prev_grad = self._module.collected_grad
            new_grad = param.grad.detach().cpu()
            
            if self._grad_acc_mode == 'mean_squared':
                new_grad = new_grad ** 2
            
            self._module.collected_grad = (prev_grad * prev_cnt + new_grad) / new_cnt
            self._module.collected_grad_cnt = new_cnt

        # remove the gradient to save memory
        param.grad = None

class SaveInputHook:
    def __init__(self, name: str, module: RosaLayer) -> None:
        self._name = name
        self._module = module

    def __call__(self, model, module_in, module_out):
        if not isinstance(module_in, torch.Tensor):
            if len(module_in) > 1:
                print(f'found {len(module_in)} inputs, keeping only the first one.')
            module_in = module_in[0]
        
        if hasattr(self._module, 'saved_input'):
            self._module.saved_input = module_in
        else:
            self._module.register_buffer('saved_input', module_in)
        
        print(f'saved input for {self._name}')

class ManualGradCollectorHook:
    def __init__(self, name: str, module: RosaLayer, grad_acc_mode: str) -> None:
        assert grad_acc_mode in ['mean', 'mean_squared']
        self._name = name
        self._module = module
        self._grad_acc_mode = grad_acc_mode

    def __call__(self, model, grad_in, grad_out):
        print('hook called for', self._name)
        if not isinstance(grad_out, torch.Tensor):
            if len(grad_out) > 1:
                print(f'found {len(grad_out)} grad_outs, keeping only the first one.')
            grad_out = grad_out[0]

        with torch.no_grad():
            saved_input = self._module.saved_input
            new_grad = torch.mm(
                grad_out.reshape(-1, grad_out.shape[-1]).T,
                saved_input.reshape(-1, saved_input.shape[-1]),
            )
            if isinstance(self._module.get_base_layer(), torch.nn.Embedding):
                new_grad = new_grad.T
            if self._grad_acc_mode == 'mean_squared':
                new_grad = new_grad ** 2
            new_grad = new_grad.cpu()

            if not hasattr(self._module, 'collected_grad'):
                self._module.register_buffer('collected_grad', torch.zeros_like(new_grad))
                setattr(self._module, 'collected_grad_cnt', 0)

            prev_grad = self._module.collected_grad
            prev_cnt = getattr(self._module, 'collected_grad_cnt')

            new_cnt = prev_cnt + 1
            self._module.collected_grad = (prev_grad * prev_cnt + new_grad) / new_cnt
            self._module.collected_grad_cnt = new_cnt

            self._module.saved_input.zero_()

# https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/layer.py
class RosaLayer(BaseTunerLayer):
    adapter_layer_names = ("rosa_A", "rosa_B", "rosa_embedding_A", "rosa_embedding_B", "rosa_spa")
    other_param_names = ("r", "d", "lora_alpha", "scaling", "lora_dropout")
    
    def __init__(self, base_layer: nn.Module, impl: str, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.d = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.rosa_A = nn.ModuleDict({})
        self.rosa_B = nn.ModuleDict({})

        self.rosa_spa = nn.ModuleDict({})
        self.impl = impl

        self.rosa_embedding_A = nn.ParameterDict({})
        self.rosa_embedding_B = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        self.rosa_dummy = torch.tensor([0.], dtype=torch.bfloat16, requires_grad=True)

    def _add_dummy(self, x: torch.Tensor):
        with torch.no_grad():
            self.rosa_dummy.zero_()
            if self.rosa_dummy.device != x.device:
                self.rosa_dummy = self.rosa_dummy.to(x.device)
            self.rosa_dummy.requires_grad = True
        return x + self.rosa_dummy.to(x.dtype)

    def _get_weight_shape(self):
        if isinstance(self.get_base_layer(), torch.nn.Embedding):
            return (self.in_features, self.out_features)
        return (self.out_features, self.in_features)

    def update_layer(self, adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora):
        if r < 0:
            raise ValueError(f"`r` should be a non-negative integer value but the value passed is {r}")

        if d < 0 or d > 1:
            raise ValueError(f"`d` should be a value between 0 and 1 but the value passed is {d}")
        

        self.r[adapter_name] = r
        self.d[adapter_name] = d

        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters

        if r == 0:
            self.scaling[adapter_name] = 1.
        elif use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        
        rosa_dtype = torch.bfloat16 if rosa_dtype == 'bf16' else (torch.float16 if rosa_dtype == 'fp16' else torch.float32)
        if r > 0:
            self.rosa_A[adapter_name] = nn.Linear(self.in_features, r, bias=False, dtype=rosa_dtype)
            self.rosa_B[adapter_name] = nn.Linear(r, self.out_features, bias=False, dtype=rosa_dtype)
        else:
            self.rosa_A[adapter_name] = nn.Identity()
            self.rosa_B[adapter_name] = nn.Identity()

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        
        device = None
        dtype = None
        weight_shape = self._get_weight_shape()
        
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    dtype = weight.dtype
                device = weight.device
                break
        
        assert None not in [device, weight_shape], "weight or qweight should be available"

        if d > 0:
            self.rosa_spa[adapter_name] = SparseLinear(
                density=d,
                shape=weight_shape,
                store_transpose=spa_store_transpose if self.impl == 'spmm' else False, # 'sp_add' does not requires the transpositions
                dtype=rosa_dtype
            )
        else:
            self.rosa_spa[adapter_name] = nn.Identity()

        self.to(device)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if self.r[adapter_name] <= 0:
            return

        if adapter_name in self.rosa_A.keys():
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.rosa_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.rosa_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.rosa_B[adapter_name].weight)
        if adapter_name in self.rosa_embedding_A.keys():
            nn.init.zeros_(self.rosa_embedding_A[adapter_name])
            nn.init.normal_(self.rosa_embedding_B[adapter_name])

    def loftq_init(self, adapter_name):
        if self.r[adapter_name] <= 0:
            assert False, "LoftQ is only supported for r > 0"
            return

        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, rosa_A, rosa_B = loftq_init(weight, **kwargs)
        if adapter_name in self.rosa_A.keys():
            self.rosa_A[adapter_name].weight.data = rosa_A
            self.rosa_B[adapter_name].weight.data = rosa_B
        if adapter_name in self.rosa_embedding_A.keys():
            self.rosa_embedding_A[adapter_name].data = rosa_A
            self.rosa_embedding_B[adapter_name].data = rosa_B
        self.get_base_layer().weight.data = qweight

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return

        if self.r[adapter] <= 0:
            return

        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.rosa_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.rosa_A.keys():
                continue

            if scale is None:
                if self.r[active_adapter] > 0:
                    self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def set_spa_mask(self, mask):
        assert len(self.active_adapters) <= 1, 'at most one RoSA adapter is supported for now'
        assert len(self.active_adapters) == 1, 'set_spa_mask was called but no active adapter found'
        adapter = self.active_adapters[0]

        assert adapter in self.rosa_spa, 'set_spa_mask was called for an adapter that does not exist'
        spa_module = self.rosa_spa[adapter]
        assert spa_module is not None, 'set_spa_mask was called while there is no spa_module'

        spa_module.set_mask(mask)

    def _spa_exists(self, adapter):
        if adapter not in self.d or self.d[adapter] <= 0:
            return False
        if not self.rosa_spa[adapter].exists():
            return False
        return True

    def _convert_spa_to_dense(self, adapter):
        assert self._spa_exists(adapter), 'spa does not exist, but _convert_spa_to_dense was called'
        return self.rosa_spa[adapter].to_dense()

    def find_weight(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        for weight_name in ("weight", "qweight"):
            weight = getattr(base_layer, weight_name, None)
            if weight is not None:
                return weight

    def set_lora_requires_grad(self, req_grad: bool):
        for active_adapter in self.active_adapters:
            for param_dict in [self.rosa_embedding_A, self.rosa_embedding_B]:
                if active_adapter not in param_dict:
                    continue
                param = param_dict[active_adapter]
                param.requires_grad = req_grad
            
            for module_dict in [self.rosa_A, self.rosa_B]:
                if active_adapter not in module_dict:
                    continue
                module = module_dict[active_adapter]
                if not hasattr(module, "weight"):
                    continue
                module.weight.requires_grad = req_grad

    def set_spa_requires_grad(self, req_grad: bool):
        for active_adapter in self.active_adapters:
            if active_adapter not in self.rosa_spa:
                continue
            module = self.rosa_spa[active_adapter]
            module.values.requires_grad = req_grad


class Linear(nn.Module, RosaLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        d: float = 0.0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        impl: str = 'auto',
        spa_store_transpose: bool = True,
        rosa_dtype: str = 'bf16',
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if impl == 'auto':
            impl = 'sp_add'
        RosaLayer.__init__(self, base_layer, impl, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.rosa_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.rosa_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        if self.r[adapter] > 0:
            device = self.rosa_B[adapter].weight.device
            dtype = self.rosa_B[adapter].weight.dtype
        else:
            device = self.rosa_spa[adapter].values.device
            dtype = self.rosa_spa[adapter].values.dtype

        output_tensor = None
        if self.r[adapter] > 0:
            cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

            weight_A = self.rosa_A[adapter].weight
            weight_B = self.rosa_B[adapter].weight

            if cast_to_fp32:
                weight_A = weight_A.float()
                weight_B = weight_B.float()

            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

                # cast back the weights
                self.rosa_A[adapter].weight.data = weight_A.to(dtype)
                self.rosa_B[adapter].weight.data = weight_B.to(dtype)
        
        if self._spa_exists(adapter):
            spa_dense = self._convert_spa_to_dense(adapter).to(dtype)
            if output_tensor is None:
                output_tensor = spa_dense
            else:
                output_tensor += spa_dense

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            assert len(self.active_adapters) == 1, 'rosa only supports precisely one adapter'
            active_adapter = self.active_adapters[0]
            assert active_adapter in self.rosa_A.keys()

            if self.r[active_adapter] == 0 and not self._spa_exists(active_adapter):
                x = self._add_dummy(x)

            if self.impl == 'spmm' or not self._spa_exists(active_adapter): # sp_add implementation is suboptimal when spa does not exist
                result = self.base_layer(x, *args, **kwargs)
                
                if self.r[active_adapter] > 0:
                    rosa_A = self.rosa_A[active_adapter]
                    rosa_B = self.rosa_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = x.to(rosa_A.weight.dtype)
                    result += rosa_B(rosa_A(dropout(x))) * scaling

                if self._spa_exists(active_adapter):
                    spa_module = self.rosa_spa[active_adapter]
                    # x = x.to(spa_module.values.dtype)
                    result += spa_module(x)
            else:
                assert self.impl == 'sp_add', f'unknown rosa implementation {self.impl}'
                dropout = self.lora_dropout[active_adapter]
                dropout_rate = dropout.p if isinstance(dropout, nn.Dropout) else 0
                scaling = self.scaling[active_adapter]
                result = RoSALinearFunction.apply(
                    x,
                    self.get_base_layer(),
                    getattr(self.rosa_A[active_adapter], 'weight', None),
                    getattr(self.rosa_B[active_adapter], 'weight', None),
                    getattr(self.rosa_spa[active_adapter], 'values', None),
                    getattr(self.rosa_spa[active_adapter], 'row_offs', None),
                    getattr(self.rosa_spa[active_adapter], 'row_idx', None),
                    getattr(self.rosa_spa[active_adapter], 'col_idx', None),
                    scaling,
                    dropout_rate,
                    self.training
                )
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rosa." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    rosa_config: RosaConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(rosa_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = rosa_config.fan_in_fan_out = False
        kwargs.update(rosa_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module

# https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/model.py
class RosaModel(BaseTuner):
    
    prefix: str = "rosa_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        self._spa_activated = False

    def _check_new_adapter_config(self, config: RosaConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(rosa_config, key):
        return check_target_module_exists(rosa_config, key)

    def _create_and_replace(
        self,
        rosa_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        pattern_keys = list(chain(rosa_config.rank_pattern.keys(), rosa_config.density_pattern.keys(), rosa_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(f".*\.{key}$", current_key), pattern_keys), current_key)
        r = rosa_config.rank_pattern.get(target_name_key, rosa_config.r)
        d = rosa_config.density_pattern.get(target_name_key, rosa_config.d)
        alpha = rosa_config.alpha_pattern.get(target_name_key, rosa_config.lora_alpha)

        kwargs = {
            "r": r,
            "d": d,
            "lora_alpha": alpha,
            "lora_dropout": rosa_config.lora_dropout,
            "impl": rosa_config.impl,
            "spa_store_transpose": rosa_config.spa_store_transpose,
            "rosa_dtype": rosa_config.rosa_dtype,
            "fan_in_fan_out": rosa_config.fan_in_fan_out,
            "init_lora_weights": rosa_config.init_lora_weights,
            "use_rslora": rosa_config.use_rslora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            assert False, "RoSA does not support GPTQ quantization"

        if isinstance(target, RosaLayer):
            target.update_layer(
                adapter_name,
                r=r,
                d=d,
                lora_alpha=alpha,
                lora_dropout=rosa_config.lora_dropout,
                spa_store_transpose=rosa_config.spa_store_transpose,
                rosa_dtype=rosa_config.rosa_dtype,
                init_lora_weights=rosa_config.init_lora_weights,
                use_rslora=rosa_config.use_rslora,
            )
        else:
            new_module = self._create_new_module(rosa_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "rosa_only":
                for m in model.modules():
                    if isinstance(m, RosaLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(rosa_config, adapter_name, target, **kwargs):
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend([dispatch_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, rosa_config=rosa_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        for module in self.model.modules():
            if isinstance(module, RosaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[List[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge RoSA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type="svd",
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
    ) -> None:
        assert False, "RoSA does not support add_weighted_adapter"

    def delete_adapter(self, adapter_name: str) -> None:
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, RosaLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> torch.nn.Module:
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        return self._unload_and_optionally_merge(merge=False)

    def _find_mask(self, masks, target_key):
        for key, mask in masks.items():
            if target_key in key:
                return mask

    def set_spa_masks(self, masks):
        for name, module in self.named_modules():
            if not isinstance(module, RosaLayer):
                continue
            mask = self._find_mask(masks, name)
            assert mask is not None, f'missing key {name} in masks.'
            module.set_spa_mask(mask)
        print('spa masks set.')
        self._spa_activated = True

    @property
    def spa_activated(self) -> bool:
        return self._spa_activated
    
# https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/rosa_functions.py
class RoSALinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X, W_module, LA, LB, S_val, S_row_offs, S_row_idx, S_col_idx, lora_scaling, lora_dropout_rate, training):
        # assert S_val is not None, 'sp_add implementation of RoSA is suboptimal if there is no sparse adapter, please switch to the spmm implementation.'

        input_shape = X.shape
        X = X.reshape(-1, X.shape[-1])

        needs_4bit_deq = False
        orig_W = W_module.weight if hasattr(W_module, 'weight') else W_module.qweight
        b = W_module.bias if hasattr(W_module, 'bias') else None
        if orig_W.dtype in [torch.bfloat16, torch.float16, torch.float32]:
            W = orig_W.to(X.dtype)
        else:
            assert isinstance(W_module, bnb.nn.Linear4bit), 'only [bf16, fp16, fp32] and 4bit quantization are supported in the sp_add implementation of RoSA. Change the implementation to spmm.'
            needs_4bit_deq = True
            W = bnb.functional.dequantize_4bit(orig_W.data, orig_W.quant_state).to(X.dtype)
        
        if S_val is None:
            O = torch.mm(X, W.T)
        else:
            O = torch.mm(X, csr_add(S_val, S_row_offs, S_row_idx, S_col_idx, W).T)

        if b is not None:
            O += b.to(X.dtype).unsqueeze(0)
        
        keep_prob = None
        D = None # the dropout mask
        if LA is not None:
            if training:
                keep_prob = 1 - lora_dropout_rate
                D = torch.rand_like(X) < keep_prob
                O += lora_scaling * torch.mm(torch.mm((X * D) / keep_prob, LA.T), LB.T)
            else:
                O += lora_scaling * torch.mm(torch.mm(X, LA.T), LB.T)

        ctx.save_for_backward(X, orig_W, LA, LB, S_val, S_row_offs, S_row_idx, S_col_idx, D)
        ctx.needs_4bit_deq = needs_4bit_deq
        ctx.input_shape = input_shape
        ctx.lora_scaling = lora_scaling
        ctx.keep_prob = keep_prob
        
        return O.reshape(*input_shape[:-1], O.shape[-1])

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dO):
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
        return dX, None, dLA, dLB, dS_val, None, None, None, None, None, None
    
# https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/scheduler.py
try:
    from composer.core import Algorithm, Event
    COMPOSER_ALG_CLASS = Algorithm
    COMPOSER_EVENT_CLASS = Event
except ImportError:
    COMPOSER_ALG_CLASS = object
    COMPOSER_EVENT_CLASS = None


class RosaScheduler(TrainerCallback, COMPOSER_ALG_CLASS):
    def __init__(self, model: RosaModel) -> None:
        COMPOSER_ALG_CLASS.__init__(self)
        TrainerCallback.__init__(self)

        self._model = model

        config = model.peft_config
        assert len(config) == 1 and 'default' in config, 'only one default adapter is supported for now'
        config = config['default']

        self._mask_load_path = getattr(config, 'mask_load_path', None)
        self._mask_save_path = getattr(config, 'mask_save_path', None)
        self._spa_num_grads = getattr(config, 'spa_num_grads', 1)
        self._grad_acc_mode = getattr(config, 'grad_acc_mode', 'mean_squared')
        self._terminate_after_mask_generation = getattr(config, 'terminate_after_mask_generation', False)
        
        self._d = getattr(config, 'd', 0.)
        self._r = getattr(config, 'r', 0)

        assert None in [self._mask_load_path, self._mask_save_path], 'at least one of mask_save_path and mask_load_path has to be none.'
        if self._d > 0:
            if self._terminate_after_mask_generation:
                assert self._mask_save_path is not None
                assert self._mask_load_path is None

            if self._mask_load_path is not None:
                self._set_spa_masks(torch.load(self._mask_load_path))

        schedule_name = getattr(config, 'schedule', None)
        self._schedule = self._create_schedule(schedule_name)

        self._step = 0
        self._handles = []
    
    def _create_schedule(self, schedule_name: str) -> List[dict]:
        assert schedule_name is not None, "RoSA schedule has to be specified"

        if schedule_name in ['default', 'df']:
            return self._create_schedule('wl0')
        
        elif schedule_name == 'spa_only':
            assert self._d > 0, 'spa_only schedule requires density > 0'
            return self._generate_spa_schedule(self._mask_load_path is None)
        
        elif schedule_name == 'lora_only':
            assert self._d == 0, 'lora_only schedule requires density = 0'
            return self._generate_lora_schedule()
        
        elif schedule_name.startswith('wl'): # wl64 or wl224
            assert schedule_name == 'wl0' or self._d > 0, 'wl schedule requires density > 0'
            lora_warmup_steps = int(schedule_name.split('wl')[-1])
            return self._generate_wl_schedule(lora_warmup_steps, self._mask_load_path is None)
        else:
            assert False, f"RoSA schedule {schedule_name} is not implemented (df and ws schedules will be implemented later)."

    def _generate_spa_schedule(self, grad_colletion_needed: bool) -> List[dict]:
        schedule = []
        if grad_colletion_needed:
            schedule.append({'agenda': ['grad_collection'], 'end': self._spa_num_grads})
        schedule.append({'agenda': ['spa'], 'end': None})
        return schedule

    def _generate_lora_schedule(self) -> List[dict]:
        schedule = [{'agenda': ['lora'], 'end': None}]
        return schedule
    
    def _generate_wl_schedule(self, warmup: int, grad_colletion_needed: bool) -> List[dict]:
        schedule = []
        if warmup > 0:
            schedule.append({'agenda': ['lora'], 'end': warmup})
        if grad_colletion_needed:
            schedule.append({'agenda': ['lora', 'grad_collection'], 'end': warmup + self._spa_num_grads})
        schedule.append({'agenda': ['lora', 'spa'], 'end': None})
        return schedule

    def _get_agenda(self, step: int) -> List:
        for item in self._schedule:
            if item['end'] is None or step < item['end']:
                return item['agenda']
        assert False, f"no agenda for step {step}"

    def _get_current_agenda(self) -> List:
        return self._get_agenda(self._step)

    def _get_prev_agenda(self) -> List:
        return self._get_agenda(self._step - 1) if self._step > 0 else None

    def _get_next_agenda(self) -> List:
        return self._get_agenda(self._step + 1)
    
    def _set_spa_masks(self, masks: Dict[str, torch.Tensor]) -> None:
        self._model.set_spa_masks(masks)
    
    def match(self, event, state):
        if COMPOSER_EVENT_CLASS is None:
            return False
        return event in [COMPOSER_EVENT_CLASS.BEFORE_TRAIN_BATCH, COMPOSER_EVENT_CLASS.AFTER_TRAIN_BATCH]

    def apply(self, event, state, logger):
        if COMPOSER_EVENT_CLASS is None:
            return
        
        if event == COMPOSER_EVENT_CLASS.BEFORE_TRAIN_BATCH:
            self._on_step_begin()
        elif event == COMPOSER_EVENT_CLASS.AFTER_TRAIN_BATCH:
            self._on_step_end()

    def on_step_begin(self, args, state, control, **kwargs):
        self._on_step_begin()

    def on_step_end(self, args, state, control, **kwargs):
        self._on_step_end()

    @torch.no_grad()
    def _on_step_begin(self):
        agenda = self._get_current_agenda()
        print('AGENDA', agenda)

        for _, p in self._model.named_parameters():
            p.requires_grad = False

        if self._mask_load_path is not None and not self._model.spa_activated:
            print('loading masks')
            masks = torch.load(self._mask_load_path)
            self._set_spa_masks(masks) # this activates spa

        for name, module in self._model.named_modules():
            if not isinstance(module, RosaLayer):
                continue

            weight = module.find_weight()
            if 'grad_collection' in agenda and not self._model.spa_activated:
                handle1 = module.register_forward_hook(SaveInputHook(name, module))
                handle2 = module.register_full_backward_hook(ManualGradCollectorHook(name, module, self._grad_acc_mode))
                self._handles.append(handle1)
                self._handles.append(handle2)
            else:
                if weight.is_floating_point:
                    weight.requires_grad = False
            
            module.set_lora_requires_grad('lora' in agenda)
            
            if self._model.spa_activated:
                module.set_spa_requires_grad('spa' in agenda)

    @torch.no_grad()
    def _on_step_end(self):
        agenda = self._get_current_agenda()
        next_agenda = self._get_next_agenda()

        if not self._model.spa_activated and 'grad_collection' in agenda and 'grad_collection' not in next_agenda:
            print('finished collecting gradients')
            self._generate_masks_and_activate_spa(self._model)

        for handle in self._handles:
            handle.remove()
        
        self._handles = []
        self._step += 1
    
    @torch.no_grad()
    def _grad_to_mask_fn(self, grad):
        idx = torch.topk(torch.abs(grad.flatten()).float(), int(self._d * grad.numel()), sorted=False).indices
        mask = torch.zeros_like(grad.flatten())
        mask.scatter_(0, idx, 1.)
        mask = mask.reshape_as(grad).bool()
        return mask

    @torch.no_grad()
    def _generate_masks_and_activate_spa(self, model):
        print('generating masks and activating spa')
        assert self._d > 0, 'mask generation requires spa density to be > 0'

        masks = {}
        for name, module in model.named_modules():
            if not isinstance(module, RosaLayer):
                continue

            assert hasattr(module, 'collected_grad'), 'target module must have a collected_grad for mask generation, something is wrong!'
            print(f'generating spa mask for {name} with {module.collected_grad_cnt} grads.')

            masks[name] = self._grad_to_mask_fn(module.collected_grad)
            delattr(module, 'collected_grad')
            delattr(module, 'collected_grad_cnt')
            if hasattr(module, 'saved_input'):
                delattr(module, 'saved_input')
        
        if self._mask_save_path is not None:
            print('saving the masks...')
            torch.save(masks, self._mask_save_path)
            print('masks saved.')
        
        if self._terminate_after_mask_generation:
            print('Done. halting...')
            raise SystemExit()
        else:
            self._set_spa_masks(masks)
            
# https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/spa_functions.py
class SpMMFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, A_val, A_row_offsets, A_row_indices, A_col_indices, B, M, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices):
        ctx.save_for_backward(A_val, A_row_offsets, A_row_indices, A_col_indices, B, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices)
        C = spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, M)
        return C

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dLdC):
        A_val, A_row_offsets, A_row_indices, A_col_indices, B, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices = ctx.saved_tensors
        dLdC = dLdC.contiguous()
        
        dLdA_val = sddmm(A_row_offsets, A_row_indices, A_col_indices, dLdC, B)

        if AT_val is None:
            AT_val, AT_row_offsets, AT_col_indices = csr_transpose(A_val, A_row_offsets, A_col_indices, dLdC.shape[0], B.shape[0])
            AT_row_indices = torch.argsort(-1 * torch.diff(AT_row_offsets)).int()
            
        dLdB = spmm(AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, dLdC, B.shape[0])
        return dLdA_val.to(A_val.dtype), None, None, None, dLdB.to(B.dtype), None, None, None, None, None
        
class SpMMTFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, B, M, A_val, A_row_offsets, A_row_indices, A_col_indices): # A: (M, K), AT: (K, M)
        if A_val is None:
            A_val, A_row_offsets, A_col_indices = csr_transpose(AT_val, AT_row_offsets, AT_col_indices, B.shape[0], M)
            A_row_indices = torch.argsort(-1 * torch.diff(A_row_offsets)).int()

        ctx.save_for_backward(AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, B)
        C = spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, M)
        return C

    @staticmethod
    @once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dLdC):
        AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, B = ctx.saved_tensors
        dLdC = dLdC.contiguous()
        
        dLdAT_val = sddmm(AT_row_offsets, AT_row_indices, AT_col_indices, B, dLdC)
        dLdB = spmm(AT_val, AT_row_offsets, AT_row_indices, AT_col_indices, dLdC, B.shape[0])
        return dLdAT_val.to(AT_val.dtype), None, None, None, dLdB.to(B.dtype), None, None, None, None, None
    

class SparseLinear(nn.Module):
    def __init__(self, density, shape, store_transpose=False, dtype=torch.bfloat16):
        super(SparseLinear, self).__init__()
        
        self.shape = shape
        self.store_transpose = store_transpose

        nnz = int(density * np.prod(shape))
        self.values = nn.Parameter(torch.zeros((nnz, ), dtype=dtype))

        self.register_buffer('row_offs', torch.zeros((shape[0] + 1, ), dtype=torch.int32))
        self.register_buffer('row_idx', torch.zeros((shape[0], ), dtype=torch.int16))
        self.register_buffer('col_idx', torch.zeros((nnz, ), dtype=torch.int16))

        if self.store_transpose:
            self.register_buffer('tr_perm', torch.zeros((nnz, ), dtype=torch.int32))
            self.register_buffer('tr_row_offs', torch.zeros((shape[1] + 1, ), dtype=torch.int32))
            self.register_buffer('tr_row_idx', torch.zeros((shape[1], ), dtype=torch.int16))
            self.register_buffer('tr_col_idx', torch.zeros((nnz, ), dtype=torch.int16))

    @torch.no_grad()
    def set_mask(self, mask):
        nnz = mask.sum().int().item()
        assert self.values.numel() == nnz, f'mask.nnz does not match the numel of spa values. mask.nnz: {nnz}, spa.values.numel: {spa_module.values.numel()}'
        assert mask.shape[0] == self.shape[0] and mask.shape[1] == self.shape[1], f'mask.shape does not match spa.shape. mask.shape: {mask.shape}, spa.shape: {self.shape}'
        
        sparse_tensor = csr_matrix(mask.cpu())
        self.row_offs = torch.tensor(sparse_tensor.indptr, dtype=torch.int32, device=self.values.device)
        self.col_idx = torch.tensor(sparse_tensor.indices, dtype=torch.int16, device=self.values.device)
        self.row_idx = torch.argsort(-1 * torch.diff(self.row_offs)).to(torch.int16)

    @torch.no_grad()
    def to_dense(self):
        assert self.exists(), 'spa.to_dense() called before spa mask is set'
        return torch.sparse_csr_tensor(
            self.row_offs.to(torch.int64),
            self.col_idx.to(torch.int64),
            self.values.data,
            size=self.shape,
            dtype=self.values.dtype,
            device=self.values.device
        ).to_dense()
    
    @torch.no_grad()
    def tr(self, none_if_not_exist=False):
        if self.store_transpose:
            if self.tr_row_offs[-1] == 0:
                tr_perm_plus_one, tr_row_offs, tr_col_idx = spops.csr_transpose(
                    torch.arange(self.values.shape[0], dtype=torch.float32, device=self.values.device) + 1,
                    self.row_offs,
                    self.col_idx,
                    *self.shape
                )
                self.tr_perm = (tr_perm_plus_one - 1).int()
                self.tr_row_offs = tr_row_offs.int()
                self.tr_col_idx = tr_col_idx.to(torch.int16)
                self.tr_row_idx = torch.argsort(-1 * torch.diff(tr_row_offs)).to(torch.int16)
            return (
                self.values.data[self.tr_perm].contiguous(), 
                self.tr_row_offs, 
                self.tr_row_idx, 
                self.tr_col_idx
            )
        else:
            if none_if_not_exist:
                return [None] * 4   
            tr_values, tr_row_offs, tr_col_idx = spops.csr_transpose(
                self.values.data,
                self.row_offs,
                self.col_idx,
                *self.shape
            )
            tr_row_idx = torch.argsort(-1 * torch.diff(tr_row_offs)).int()
            return (tr_values, tr_row_offs, tr_row_idx, tr_col_idx)

    def exists(self):
        if None in [
            self.values,
            self.row_offs,
            self.row_idx,
            self.col_idx
        ]:
            return False
        return self.row_offs[-1] != 0

    def forward(self, x):
        assert self.exists(), 'spa.forward() called before spa mask is set'
        tr_values, tr_row_offs, tr_row_idx, tr_col_idx = self.tr(none_if_not_exist=True)

        return SpMMFunction.apply( # only calculates grad for spa_values and x
            self.values,
            self.row_offs,
            self.row_idx,
            self.col_idx,
            x.reshape(-1, x.shape[-1]).T.contiguous(),
            self.shape[0],
            tr_values.detach() if tr_values is not None else None,
            tr_row_offs,
            tr_row_idx,
            tr_col_idx
        ).T.reshape(*x.shape[:-1], self.shape[0])


class SparseLinearT(SparseLinear):
    def forward(self, x):
        assert self.exists(), 'spa.forward() called before spa mask is set'
        tr_values, tr_row_offs, tr_row_idx, tr_col_idx = self.tr(none_if_not_exist=True)

        return SpMMTFunction.apply( # only calculates grad for spa_values and x_onehot
            self.values,
            self.row_offs,
            self.row_idx,
            self.col_idx,
            x.reshape(-1, x.shape[-1]).T.contiguous(),
            self.shape[1],
            tr_values.detach() if tr_values is not None else None,
            tr_row_offs,
            tr_row_idx,
            tr_col_idx
        ).T.reshape(*x.shape[:-1], self.shape[1])