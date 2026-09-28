#!/usr/bin/env python3
"""An explicit, dependency-light spec of TiRex2.forward(), checked against the golden activations.

Only basic torch ops (no xlstm / flashrnn / tirex2 imports) and the converted safetensors
weights are used, so every formula here is one the Java engine has to reproduce. Each stage is
compared with the matching ``act.*`` tensor of every golden case.

Usage: spec_forward.py --weights tirex2.safetensors --golden golden-cpu/
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

D, P, L, NQ = 512, 32, 74, 9
H_T = 4  # time-mixer heads (mLSTM and sLSTM)
H_V = 4  # variate-attention heads


def rms(x, w, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def head_ln(x, w, heads, eps=1e-6):
    s = x.shape
    x = x.reshape(*s[:-1], heads, -1)
    x = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + eps)
    return x.reshape(s) * w


def lin(x, W, pre, bias=False):
    y = x @ W[pre + ".weight"].T
    return y + W[pre + ".bias"] if bias else y


def residual_block(x, W, pre):
    h = F.silu(lin(x, W, pre + ".hidden_layer", True))
    return lin(h, W, pre + ".output_layer", True) + lin(x, W, pre + ".residual_layer", True)


def scaler(x):
    valid = ~torch.isnan(x)
    n = valid.sum(-1, keepdim=True)
    loc = torch.where(n > 0, torch.nansum(x, -1, keepdim=True) / n.clamp_min(1), 0.0)
    var = torch.nansum((x - loc) ** 2, -1, keepdim=True) / n.clamp_min(1)
    scale = torch.where(n > 0, var.sqrt(), 1.0).clamp_min(1e-8)
    binary = ((x == 0) | (x == 1) | ~valid).all(-1, keepdim=True)
    loc = torch.where(binary, 0.0, loc)
    scale = torch.where(binary, 1.0, scale)
    z = (x - loc) / scale
    z = torch.where(binary, z, torch.arcsinh(z))
    return z, loc, scale, binary


def mlstm_cell(x, W, pre):
    """mLSTM, parallel stabilised form. x: [B, S, D] -> [B, S, D]."""
    B, S, _ = x.shape
    q = lin(x, W, pre + ".q").reshape(B, S, H_T, -1).transpose(1, 2)  # [B,H,S,64]
    k = lin(x, W, pre + ".k").reshape(B, S, H_T, -1).transpose(1, 2)
    v = lin(x, W, pre + ".v").reshape(B, S, H_T, -1).transpose(1, 2)  # [B,H,S,128]
    o = lin(x, W, pre + ".ogate_preact")
    cap = lambda z: 15.0 * torch.tanh(z / 15.0)
    i = cap(lin(x, W, pre + ".igate_preact", True)).transpose(1, 2)  # [B,H,S]
    f = cap(lin(x, W, pre + ".fgate_preact", True)).transpose(1, 2)

    Fc = torch.cumsum(F.logsigmoid(f), -1)  # [B,H,S]: log decay from step 0 through t
    logD = Fc[..., :, None] - Fc[..., None, :] + i[..., None, :]  # [B,H,t,s]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    logD = logD.masked_fill(~causal, -math.inf)
    # the zero initial state carries stabiliser 0, decayed by Fc[t]
    m = torch.maximum(logD.amax(-1), Fc)  # [B,H,S]
    Dm = torch.exp(logD - m[..., None])
    C = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5) * Dm
    num = C @ v
    den = torch.maximum(C.sum(-1).abs(), torch.exp(-m))[..., None]
    h = num / (den + 1e-6)  # [B,H,S,128]
    h = h.transpose(1, 2).reshape(B, S, D)
    h = head_ln(h, W[pre + ".multihead_norm.weight"], H_T)
    return lin(torch.sigmoid(o) * h, W, pre + ".out_proj")


def slstm_cell(x, W, pre):
    """sLSTM recurrence (FlashRNN vanilla). x: [B, S, D] -> [B, S, D]."""
    B, S, _ = x.shape
    dh = D // H_T
    xh = x.reshape(B, S, H_T, dh)
    # slot order (i, f, z, o) is fed by the projections (fgate, igate, zgate, ogate)
    Wx = torch.stack([torch.einsum("bshd,hod->bsho", xh, W[f"{pre}.{g}.weight"]) for g in ("fgate", "igate", "zgate", "ogate")], 2)
    R = W[pre + "._recurrent_kernel_"]  # [H, P(prev), G, D]
    b = W[pre + "._bias_"]  # [H, G, D]
    y = torch.zeros(B, H_T, dh)
    c = torch.zeros_like(y)
    n = torch.zeros_like(y)
    m = torch.zeros_like(y)
    out = []
    for t in range(S):
        Ry = torch.einsum("bhp,hpgd->bghd", y, R)
        raw = Wx[:, t] + Ry + b.permute(1, 0, 2)[None]  # [B, G, H, D]
        ir, fr, zr, orw = raw.unbind(1)
        logfm = F.logsigmoid(fr) + m
        m_new = ir if bool((n == 0).all()) else torch.maximum(ir, logfm)
        ig = torch.exp(ir - m_new)
        fg = torch.exp(logfm - m_new)
        c = fg * c + ig * torch.tanh(zr)
        n = torch.maximum(fg * n + ig, torch.ones_like(n))
        y = torch.sigmoid(orw) * c / n
        m = m_new
        out.append(y)
    y = torch.stack(out, 1).reshape(B, S, D)
    return head_ln(y, W[pre + ".group_norm.weight"], H_T)


def time_mixer(x, W, pre, kind, known):
    xn = rms(x, W[pre + ".norm_lstm.weight"], 1e-6)
    cell = mlstm_cell if kind == "m" else slstm_cell
    y = cell(xn, W, pre + ".fwd_cell")
    if known.numel():
        rev = torch.flip(cell(torch.flip(xn[known], (1,)), W, pre + ".fwd_cell"), (1,))
        y = y.clone()
        y[known] = lin(torch.cat([y[known], rev], -1), W, pre + ".combination")
    x = x + y
    h = rms(x, W[pre + ".norm_ffn.weight"], 1e-6)
    ffn = F.silu(lin(h, W, pre + ".ffn.proj_up_gate")) * lin(h, W, pre + ".ffn.proj_up")
    return x + lin(ffn, W, pre + ".ffn.proj_down")


def variate_mixer(x, W, pre, group, target):
    """x: [L, R, D]; attention across rows R of the same group."""
    _, inv, counts = torch.unique(group, return_inverse=True, return_counts=True)
    keep = counts[inv] > 1
    if not keep.any():
        return x
    xin, x = x, x[:, keep]
    g, t = group[keep], target[keep]
    Lr, R = x.shape[0], x.shape[1]
    n = rms(x, W[pre + ".norm_attn.weight"], 1e-6)
    heads = lambda z: z.reshape(Lr, R, H_V, -1).permute(0, 2, 1, 3)  # [L, H, R, 128]
    eps = torch.finfo(torch.float32).eps
    Q = rms(heads(lin(n, W, pre + ".attn.WQ")), None, eps)
    K = rms(heads(lin(n, W, pre + ".attn.WK")), None, eps)
    V = heads(lin(n, W, pre + ".attn.WV"))
    allowed = (g[:, None] == g[None, :]) & ~(~t[:, None] & t[None, :])  # [query, key]
    s = (Q @ K.transpose(-1, -2)) * (Q.shape[-1] ** -0.5)
    s = s.masked_fill(~allowed, -math.inf)
    a = torch.softmax(s, -1) @ V
    x = x + lin(a.permute(0, 2, 1, 3).reshape(Lr, R, D), W, pre + ".attn.WO")
    h = rms(x, W[pre + ".norm_ffn.weight"], 1e-6)
    x = x + lin(F.silu(lin(h, W, pre + ".ffn.wi")), W, pre + ".ffn.wo")
    out = xin.clone()
    out[:, keep] = x
    return out


def forward(batch, W, recipe, future_len=320):
    x = batch["x"]
    group = batch["group_vector"]
    target = batch["target_mask"].bool()
    has_future = ~torch.isnan(x[:, -future_len:]).all(-1)
    known = torch.where(has_future & ~target)[0]

    z, loc, scale, binary = scaler(x)
    tok = z.unfold(-1, P, P)  # [R, L, P]
    mask = (~torch.isnan(tok)).float()
    h = residual_block(torch.cat([torch.nan_to_num(tok, 0.0), mask], -1), W, "input_patch_embedding")
    acts = {"input_patch_embedding": h}
    for i, kind in enumerate(recipe):
        h = time_mixer(h, W, f"stack.{i}.time_mixer", kind, known)
        acts[f"stack.{i}.time_mixer"] = h
        h = variate_mixer(h.permute(1, 0, 2), W, f"stack.{i}.variate_mixer", group, target).permute(1, 0, 2)
        acts[f"stack.{i}.variate_mixer"] = h.permute(1, 0, 2)
    h = rms(h, W["stack_out_norm.weight"], 1e-7)
    acts["stack_out_norm"] = h
    y = residual_block(h, W, "output_patch_embedding")  # [R, L, Q*P]
    acts["output_patch_embedding"] = y
    y = y.unflatten(-1, (NQ, P)).transpose(1, 2).reshape(x.shape[0], NQ, L * P)
    y = torch.where(binary[:, None], y, torch.sinh(y.clamp(-20, 20)))
    return y * scale[:, None] + loc[:, None], acts


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated subset")
    args = ap.parse_args()
    W = load_file(args.weights)
    with safe_open(args.weights, "pt") as f:
        cfg = json.loads(f.metadata()["tirex2.config"])
    recipe = ["m" if r == "mlstm_attn" else "s" for r in cfg["stack_config"]["recipe"]]
    man = json.loads((Path(args.golden) / "manifest.json").read_text())
    cases = args.cases.split(",") if args.cases else list(man["cases"])
    worst = {}
    with torch.no_grad():
        for case in cases:
            g = load_file(Path(args.golden) / f"{case}.safetensors")
            for p in range(man["cases"][case]["passes"]):
                batch = {k: g[f"pass{p}.batch.{k}"] for k in ("x", "group_vector", "target_mask")}
                out, acts = forward(batch, W, recipe)
                errs = {name: rel(a, g[f"act.pass{p}.{name}#0.out"]) for name, a in acts.items()}
                errs["model.out"] = rel(out, g[f"pass{p}.model.out"])
                stage, e = max(errs.items(), key=lambda kv: kv[1])
                print(f"{case:26s} pass{p} rows={batch['x'].shape[0]:2d}  model.out {errs['model.out']:.2e}  "
                      f"worst stage {stage} {e:.2e}", flush=True)
                for k, v in errs.items():
                    worst[k] = max(worst.get(k, 0), v)
    print("worst per stage:", {k: f"{v:.1e}" for k, v in worst.items() if not k.startswith("stack.") or k.endswith(("0.time_mixer", "11.variate_mixer"))})


if __name__ == "__main__":
    main()
