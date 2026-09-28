#!/usr/bin/env python3
"""Convert a TiRex-2 checkpoint (model.ckpt + model-config.yaml) into one safetensors file.

The PyTorch state dict stores aliases twice: ``rev_cell.*`` is the same module as
``fwd_cell.*`` (the bi-directional cell shares weights) and the sLSTM
``recurrent_kernel``/``bias`` proxies duplicate ``_recurrent_kernel_``/``_bias_``.
Only one copy of each is written. The model config is embedded as JSON in the
safetensors metadata under ``tirex2.config``.

Usage: convert_ckpt.py [--ckpt NX-AI/TiRex-2 | /path/to/dir] --out tirex2.safetensors
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def resolve(ckpt: str) -> Path:
    path = Path(ckpt).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=ckpt, allow_patterns=["model-config.yaml", "model.ckpt"]))


ALIASES = {".recurrent_kernel": "._recurrent_kernel_", ".bias": "._bias_"}


def canonical(name: str) -> str | None:
    """Return the name of the tensor ``name`` aliases, or None if it is not an alias."""
    if ".rev_cell." in name:
        return canonical(name.replace(".rev_cell.", ".fwd_cell.")) or name.replace(".rev_cell.", ".fwd_cell.")
    for proxy, stored in ALIASES.items():
        if name.endswith(".fwd_cell" + proxy):
            return name[: -len(proxy)] + stored
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="NX-AI/TiRex-2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt_dir = resolve(args.ckpt)
    config = yaml.safe_load((ckpt_dir / "model-config.yaml").read_text())
    ckpt = torch.load(ckpt_dir / "model.ckpt", map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)

    tensors, dropped = {}, []
    for name, t in state.items():
        fwd = canonical(name)
        if fwd is not None:
            assert fwd in state and torch.equal(state[fwd], t), f"{name} is not an alias of {fwd}"
            dropped.append(name)
            continue
        tensors[name] = t.detach().to(torch.float32).contiguous()

    meta = {"tirex2.config": json.dumps(config), "tirex2.source": str(ckpt_dir.name)}
    save_file(tensors, args.out, metadata=meta)
    n = sum(t.numel() for t in tensors.values())
    print(f"wrote {args.out}: {len(tensors)} tensors, {n} params ({n * 4 / 2**20:.1f} MiB), dropped {len(dropped)} aliases")


if __name__ == "__main__":
    main()
