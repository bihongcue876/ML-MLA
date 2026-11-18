import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MLATrain"))
from eval import build_from_config  # noqa: E402

path = ROOT / "models" / "gqa4" / "gqa4_42_20261117_005456" / "model.pt"
ck = torch.load(path, map_location="cpu", weights_only=True)
print("keys:", sorted(ck.keys()))
cfg = ck["config"]
print("arch:", cfg["arch"], "seq_len:", ck["seq_len"],
      "params:", format(ck["params"], ","), "tokens:", format(ck["tokens_seen"], ","))
model = build_from_config(cfg)
print("rebuild params match:", model.num_params() == ck["params"])
res = model.load_state_dict(ck["model"])
print("load_state_dict missing:", len(res.missing_keys), "unexpected:", len(res.unexpected_keys))
print("max_len in config:", cfg["max_len"], "需要 >= seq_len+64:", ck["seq_len"] + 64)
print("OK")
