# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/examples/rosa.py
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    device_map='auto',
    torch_dtype=compute_dtype,
    load_in_4bit=weight_bias_dtype == '4bit',
    quantization_config=quant_config,
    trust_remote_code=True,
    use_auth_token=True
)

rosa_config = {
    'lora_r': 16,
    'spa_d': 0.018,
    'lora_alpha': 16,
    'target_modules': 'all-linear',
    'lora_dropout': 0.05,
    'impl': 'auto',
    'spa_store_transpose': True,
    'rosa_dtype': 'bf16',
    'spa_num_grads': 1,
    'grad_acc_mode': 'mean_squared',
    'mask_load_path': None,
    'mask_save_path': './saved_masks',
    'terminate_after_mask_generation': False,
    'schedule': 'wl4',
    'lora_lr': 0.0007
}

sb = sten.SparsityBuilder()
for module_name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        weight = module_name + ".weight"
        weights_to_sparsify.append(weight)
        sb.set_weight(
            name=weight,
            initial_sparsifier=RoSASparsifier(
                config=rosa_config
            ),
            out_format=RoSATensor,
        )
        sb.set_weight_grad(
            name=weight,
            tmp_format=RoSATensor,
            out_format=RoSATensor,
        )
sb.sparsify_model_inplace(model)