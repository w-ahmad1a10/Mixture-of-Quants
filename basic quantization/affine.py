"""Affine quantization helpers for basic quantization experiments."""
import torch


def _block_params(block, bits, symmetric):
    if symmetric:
        maxabs = block.abs().max().item() if block.numel() else 0.0
        qmax = (1 << (bits - 1)) - 1
        scale = (maxabs / qmax) if (maxabs > 0 and qmax > 0) else 1.0
        zp = 0
        return scale, zp
    qmax = (1 << bits) - 1
    wmin = block.min().item() if block.numel() else 0.0
    wmax = block.max().item() if block.numel() else 0.0
    if wmax - wmin <= 0:
        return 1.0, 0
    scale = (wmax - wmin) / qmax
    zp = int(round(-wmin / scale))
    zp = max(0, min(qmax, zp))
    return scale, zp


def quantize_with_params(w, bits, symmetric, scale, zp):
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        q = (w / scale).round().clamp(-qmax, qmax)
        return q * scale
    qmax = (1 << bits) - 1
    q = (w / scale + zp).round().clamp(0, qmax)
    return (q - zp) * scale


def quantize_tensor(w, bits, symmetric, granularity, group_size=None):
    w = w.detach().clone().float()
    if w.dim() < 2:
        return w
    rows, cols = w.shape[0], w.shape[1]
    out = torch.empty_like(w)
    if granularity == "per_tensor":
        scale, zp = _block_params(w.reshape(-1), bits, symmetric)
        return quantize_with_params(w, bits, symmetric, scale, zp)
    if granularity == "per_channel":
        for r in range(rows):
            scale, zp = _block_params(w[r], bits, symmetric)
            out[r] = quantize_with_params(w[r], bits, symmetric, scale, zp)
        return out
    if granularity == "per_group":
        gs = group_size or 32
        for r in range(rows):
            for start in range(0, cols, gs):
                end = min(start + gs, cols)
                scale, zp = _block_params(w[r, start:end], bits, symmetric)
                out[r, start:end] = quantize_with_params(w[r, start:end], bits, symmetric, scale, zp)
        return out
    return w


def layer_params(weights, bits, symmetric):
    flat = torch.cat([x.reshape(-1).float() for x in weights]) if weights else torch.tensor([0.0])
    return _block_params(flat, bits, symmetric)
