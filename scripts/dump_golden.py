#!/usr/bin/env python3
"""Dump golden inputs, per-stage activations and forecasts of the NXAI TiRex-2 reference.

For each case this writes ``<out>/<case>.safetensors`` holding:
  in.target / in.past_covariates / in.future_covariates   the forecast() inputs
  batch.x / batch.group_vector / batch.target_mask       what TiRex2.forward() receives
  act.<module>#<call>.in / .out                          activations of the traced modules
  model.out                                              TiRex2.forward() output (pre-postprocessor)
  forecast                                               the final [V_t, 9, H] quantiles
and ``<out>/manifest.json`` describing every case (H, flags, tensor list).

The Java engine reads these files in its unit tests, so the inputs are identical.

Usage: dump_golden.py --out golden/ [--device cpu|cuda] [--reference-json .../reference_input.json]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from tirex2 import TimeseriesType, load_model

# Modules whose first tensor input and tensor output are recorded (matched as name suffixes).
TRACED = (
    "input_patch_embedding",
    "time_mixer",
    "time_mixer.norm_lstm",
    "time_mixer.fwd_cell",
    "time_mixer.fwd_cell.mlstm_backend",
    "time_mixer.fwd_cell.multihead_norm",
    "time_mixer.fwd_cell.group_norm",
    "time_mixer.combination",
    "time_mixer.ffn",
    "variate_mixer",
    "variate_mixer.attn",
    "variate_mixer.ffn",
    "stack_out_norm",
    "output_patch_embedding",
)


def series(n, length, seed, trend=0.0, season=24):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    rows = []
    for i in range(n):
        walk = torch.randn(length, generator=g).cumsum(0) * 0.3
        s = 2.0 * torch.sin(2 * math.pi * t / season + i)
        rows.append(10.0 + walk + s + trend * t)
    return torch.stack(rows)


def cases(reference_json):
    out = {}
    if reference_json:
        ref = json.loads(Path(reference_json).read_text())
        H = ref["H"]
        tgt = torch.tensor(ref["target"], dtype=torch.float32)
        other = torch.tensor(ref["other_target"], dtype=torch.float32)
        fut = torch.tensor(ref["future_covariates"], dtype=torch.float32)
        out["ref_univariate"] = (H, [TimeseriesType(tgt[None], None, None)], {})
        out["ref_multivariate"] = (H, [TimeseriesType(tgt[None], None, fut)], {})
        out["ref_multitarget"] = (H, [TimeseriesType(torch.stack([tgt, other]), None, None)], {})
        out["ref_univariate_signflip"] = (H, [TimeseriesType(tgt[None], None, None)], {"tta_sign_flip": True})

    out["short_univariate"] = (16, [TimeseriesType(series(1, 100, 1), None, None)], {})
    out["full_context"] = (64, [TimeseriesType(series(1, 2048, 2), None, None)], {})
    out["overlong_context"] = (320, [TimeseriesType(series(1, 3000, 3), None, None)], {})
    out["trend_diff"] = (32, [TimeseriesType(series(1, 400, 4, trend=0.5), None, None)], {})
    out["binary"] = (24, [TimeseriesType((series(1, 300, 5) > 10).float(), None, None)], {})
    nan_series = series(1, 500, 6)
    nan_series[0, 100:140] = float("nan")
    out["with_nans"] = (32, [TimeseriesType(nan_series, None, None)], {})
    out["batch_ragged"] = (48, [TimeseriesType(series(1, n, 10 + i), None, None) for i, n in enumerate((64, 300, 1024, 2048))], {})
    out["past_and_future_cov"] = (
        48,
        [TimeseriesType(series(1, 600, 20), series(2, 600, 21), series(2, 600 + 48, 22))],
        {},
    )
    out["mixed_batch_cov"] = (
        40,
        [
            TimeseriesType(series(2, 500, 30), None, series(1, 540, 31)),
            TimeseriesType(series(1, 800, 32), series(3, 800, 33), None),
            TimeseriesType(series(1, 200, 34), None, None),
        ],
        {},
    )
    return out


class Tracer:
    def __init__(self, model):
        self.acts = defaultdict(list)
        self.handles = []
        for name, module in model.named_modules():
            if name and any(name == t or name.endswith("." + t) or name.split(".", 2)[-1] == t for t in TRACED):
                self.handles.append(module.register_forward_hook(self._hook(name)))
        self.handles.append(model.register_forward_hook(self._model_hook, with_kwargs=True))
        self.batches, self.outputs = [], []

    def _hook(self, name):
        def hook(module, args, output):
            x = next((a for a in args if torch.is_tensor(a)), None)
            y = output[0] if isinstance(output, tuple) else output
            self.acts[name].append((x, y))

        return hook

    def _model_hook(self, module, args, kwargs, output):
        self.batches.append({k: v for k, v in args[0].items() if torch.is_tensor(v)})
        self.outputs.append(output)

    def reset(self):
        self.acts.clear()
        self.batches.clear()
        self.outputs.clear()


def cpu(t):
    return t.detach().to("cpu").clone().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="NX-AI/TiRex-2")
    ap.add_argument("--reference-json", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fm = load_model(args.ckpt, device=args.device)
    tracer = Tracer(fm.model)
    manifest = {"device": args.device, "torch": torch.__version__, "cases": {}}

    for case, (H, tss, kwargs) in cases(args.reference_json).items():
        tracer.reset()
        with torch.no_grad():
            forecasts = fm.forecast(tss, prediction_length=H, output_type="torch", **kwargs)

        tensors = {}
        for i, ts in enumerate(tss):
            tensors[f"in.{i}.target"] = cpu(ts.target)
            if ts.past_covariates is not None:
                tensors[f"in.{i}.past_covariates"] = cpu(ts.past_covariates)
            if ts.future_covariates is not None:
                tensors[f"in.{i}.future_covariates"] = cpu(ts.future_covariates)
            tensors[f"forecast.{i}"] = cpu(forecasts[i]).float()
        for p, (batch, out) in enumerate(zip(tracer.batches, tracer.outputs)):
            for k, v in batch.items():
                tensors[f"pass{p}.batch.{k}"] = cpu(v).float() if v.dtype == torch.bool else cpu(v)
            tensors[f"pass{p}.model.out"] = cpu(out)
        n_pass = max(1, len(tracer.batches))
        for name, calls in tracer.acts.items():
            per_pass = len(calls) // n_pass
            for c, (x, y) in enumerate(calls):
                tag = f"act.pass{c // per_pass}.{name}#{c % per_pass}"
                if x is not None:
                    tensors[tag + ".in"] = cpu(x).float()
                if torch.is_tensor(y):
                    tensors[tag + ".out"] = cpu(y).float()

        save_file(tensors, out_dir / f"{case}.safetensors")
        manifest["cases"][case] = {
            "H": H,
            "num_series": len(tss),
            "passes": n_pass,
            "kwargs": kwargs,
            "tensors": sorted(tensors),
        }
        print(f"{case}: {len(tensors)} tensors, passes={n_pass}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
