# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/tests/test_training.py#L142C9-L146C66
sparse_config = {}
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        sparse_config[name] = SemiSparseLinear
swap_linear_with_semi_sparse_linear(model, sparse_config)