# https://github.com/spcl/sten/blob/a1c4c97056fa633b4f6a0b54bb9039f4e9758c01/tests/test_spqr.py
quantized_model = torch.load(quantized_model_path)
quantized_model.to(device)

weights_to_sparsify = []
sb = sten.SparsityBuilder()
for module_name, module in model.named_modules():
    if module_name == '':
        continue
    qlm = rgetattr(quantized_model, module_name)
    if isinstance(qlm, spqr.QuantizedLinear):
        weight = module_name + ".weight"
        weights_to_sparsify.append(weight)
        sb.set_weight(
            name=weight,
            initial_sparsifier=SpQRSparsifier(
                m=qlm.m,
                n=qlm.n,
                bits=qlm.bits,
                beta1=qlm.beta1,
                beta2=qlm.beta2,
                dense_weights=qlm.dense_weights,
                row_offsets=qlm.row_offsets,
                col_vals=qlm.col_vals,
                in_perm=qlm.in_perm,
            ),
            out_format=SpQRTensor,
        )
sb.sparsify_model_inplace(model)