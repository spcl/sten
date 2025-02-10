# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/tests/test_sparse_marlin.py
sb = sten.SparsityBuilder()
for module_name, module in model.named_modules():
    if isinstance(module, torch.nn.modules.linear.Linear):
        weight = module_name + ".weight"
        sb.set_weight(
            name=weight,
            initial_sparsifier=SparseMarlinSparsifier(),
            out_format=SparseMarlinTensor,
        )
model = sb.sparsify_model_inplace(model)