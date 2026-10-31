"""Adam(W) moment buffers in page-locked host memory between optimizer steps (v15).

The LoRA adapter's exp_avg / exp_avg_sq (2 x 2.18 M fp32 values, 17 MB) are only read by the
optimizer step, yet the GPU would hold them through every forward and backward. After each step
(a step post-hook) they are copied into one pinned host buffer on the current stream and the GPU
tensors are released; before the next step (a step pre-hook) one copy brings them back into one
GPU buffer whose views become the state tensors again. The optimizer itself (PyTorch's AdamW, on
the GPU, same kernels) computes exactly as before: the values it reads are the values it wrote.

Only for torch.optim.Adam / AdamW whose moment buffers are fp32 CUDA tensors shaped like their
parameters; anything else is left alone. A state_dict() call waits for the copies first.
"""

import torch

_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


class _Offload:
    def __init__(self, opt, device):
        self.opt, self.device = opt, device
        self.recs = None          # [(state dict, key, offset, numel, shape)]
        self.host = None
        self.gpu = None
        self.note = "installed"

    def _build(self):
        recs, at = [], 0
        for group in self.opt.param_groups:
            for p in group["params"]:
                st = self.opt.state.get(p)
                if not st:
                    continue
                for k in _KEYS:
                    t = st.get(k)
                    if (isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float32
                            and t.shape == p.shape and t.is_contiguous()):
                        recs.append((st, k, at, t.numel(), t.shape))
                        at += -(-t.numel() // 64) * 64          # 256-byte aligned views
        if not recs:
            return False
        try:
            from . import hostmem
            host = hostmem.alloc(at * 4).view(torch.float32)
        except Exception:  # noqa: BLE001 - PyTorch's pinned allocator
            host = torch.empty(at, dtype=torch.float32, pin_memory=True)
        self.recs, self.host = recs, host
        self.note = f"{len(recs)} moment buffers ({at * 4 / 2**20:.1f} MiB) in pinned host memory"
        return True

    @torch.no_grad()
    def post(self, opt, args, kwargs):
        if self.recs is None and not self._build():
            return
        if self.gpu is not None:
            self.host.copy_(self.gpu, non_blocking=True)
        else:                                   # first step: the states the optimizer created
            for st, k, o, n, shape in self.recs:
                self.host[o:o + n].view(shape).copy_(st[k], non_blocking=True)
        for st, k, o, n, shape in self.recs:
            st[k] = self.host[o:o + n].view(shape)
        self.gpu = None                          # freed in stream order after the copy

    @torch.no_grad()
    def pre(self, opt, args, kwargs):
        if self.recs is None:
            return
        g = torch.empty(self.host.numel(), dtype=torch.float32, device=self.device)
        g.copy_(self.host, non_blocking=True)
        for st, k, o, n, shape in self.recs:
            st[k] = g[o:o + n].view(shape)
        self.gpu = g

    def sync(self, *args, **kwargs):
        if self.recs is not None:
            torch.cuda.synchronize(self.device)


def install(optimizer, device):
    """Hook the trainer's optimizer (accelerate's wrapper or the torch optimizer). Returns a note."""
    inner = getattr(optimizer, "optimizer", optimizer)
    if type(inner) not in (torch.optim.AdamW, torch.optim.Adam):
        return f"{type(inner).__name__}: optimizer states left on the GPU"
    if getattr(inner, "_soup_offload", None) is not None:
        return inner._soup_offload.note
    if not hasattr(inner, "register_state_dict_pre_hook"):
        return "torch without optimizer state_dict hooks: optimizer states left on the GPU"
    off = _Offload(inner, device)
    inner.register_step_pre_hook(off.pre)
    inner.register_step_post_hook(off.post)
    inner.register_state_dict_pre_hook(off.sync)
    inner._soup_offload = off
    return "Adam moments to pinned host memory after each step"
