# https://github.com/IST-DASLab/Sparse-Marlin/blob/69c618a0a16084a0c12d91d0e5a42b69c55ccae2/marlin/__init__.py
def mul_2_4(A, B, meta, C, s, workspace, thread_k=-1, thread_m=-1, sms=-1, max_par=16):
    marlin_cuda.mul_2_4(A, B, meta, C, s, workspace, thread_k, thread_m, sms, max_par)


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


class Layer_2_4(nn.Module):
    
    def __init__(self, infeatures, outfeatures, groupsize=-1):
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError(
                "`infeatures` must be divisible by 64 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer(
            "B", torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int)
        )
        self.register_buffer(
            "meta", torch.empty((self.n, self.k // 16), dtype=torch.int16)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.half)
        )
        self.register_buffer(
            "workspace",
            torch.zeros(
                self.n // 128 * 16, dtype=torch.int32, device=torch.device("cuda:0")
            ),
            persistent=False,
        )

    def forward(self, A):
        C = torch.empty(
            A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        )

        mul_2_4(
            A.view((-1, A.shape[-1])),
            self.B,
            self.meta,
            C.view((-1, C.shape[-1])),
            self.s,
            self.workspace,
        )
        return C

    def pack(self, linear, scales, trans=False):
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        if trans:
            perm, scale_perm, scale_perm_single = (
                _perm_2_4,
                _scale_perm_2_4,
                _scale_perm_single_2_4,
            )
        else:
            perm, scale_perm, scale_perm_single = _perm, _scale_perm, _scale_perm_single
        tile = 16
        maxq = 2**4 - 1
        s = scales
        w = linear.weight.data
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)

        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]

        mask = mask_creator(w.T).cuda().bool()
        w = mask * w.T
        w, meta = sparse_semi_structured_from_dense_cutlass(w)
        w = w.t()
        self.k = self.k // 2
        self.groupsize = self.groupsize // 2

        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i

        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
        self.meta[:, :] = meta.to(self.meta.device)


def replace_linear(module, name_filter=lambda n: True, groupsize=-1, name=""):
    if isinstance(module, Layer_2_4):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if isinstance(tmp, nn.Linear) and name_filter(name1):
            setattr(
                module,
                attr,
                Layer_2_4(tmp.in_features, tmp.out_features, groupsize=groupsize),
            )
    for name1, child in module.named_children():
        replace_linear(
            child,
            name_filter,
            groupsize=groupsize,
            name=name + "." + name1 if name != "" else name1,
        )