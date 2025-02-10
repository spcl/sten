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

config = RosaConfig(
    r=rosa_config['lora_r'],
    d=rosa_config['spa_d'],
    lora_alpha=rosa_config.get('lora_alpha', 16),
    target_modules=rosa_config.get('target_modules', 'all-linear'),
    lora_dropout=rosa_config.get('lora_dropout', 0.05),
    impl=rosa_config.get('impl', 'auto'),
    spa_store_transpose=rosa_config.get('spa_store_transpose', True),
    rosa_dtype=rosa_config.get('rosa_dtype', True),
    spa_num_grads=rosa_config.get('spa_num_grads', 1),
    grad_acc_mode=rosa_config.get('grad_acc_mode', 'mean_squared'),
    mask_load_path=rosa_config.get('mask_load_path', None),
    mask_save_path=rosa_config.get('mask_save_path', None),
    terminate_after_mask_generation=rosa_config.get('terminate_after_mask_generation', False),
    schedule=rosa_config.get('schedule', 'wl4'),
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)