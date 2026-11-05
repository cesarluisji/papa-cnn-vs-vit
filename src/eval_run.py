"""Re-evalúa una corrida ya entrenada (best.pt + history.csv) y escribe sus salidas, sin reentrenar.

Uso (desde la raíz):  python src/eval_run.py results/papa_resnet50_pre_s0
Útil si la corrida se interrumpió después de entrenar (p.ej. se colgó al evaluar en test).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES, build_model
from train import CSVDataset, evaluar_y_guardar, transforms

d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
hist = pd.read_csv(d / "history.csv").to_dict("records")
clases = CLASES[cfg["cultivo"]]
device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
df = pd.read_csv(cfg["split_csv"])
_, tf_ev = transforms(cfg["img_size"])
ds = {s: CSVDataset(df[df.split == s].reset_index(drop=True), clases, tf_ev) for s in ["val", "test"]}
model = build_model(cfg["model"], len(clases), pretrained=False).to(device)
model.load_state_dict(torch.load(d / "best.pt", map_location=device, weights_only=True))
t_train = sum(h["epoch_s"] for h in hist)
evaluar_y_guardar(model, ds, cfg, d, cfg["run"], hist, t_train, device, cfg["amp"] and device == "cuda", cfg["seed"])
m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
m["gpu_peak_mb"] = None  # el pico de entrenamiento no se registró (solo se re-evaluó)
m["nota"] = "re-evaluada con src/eval_run.py: la corrida se colgó al crear los workers de test"
(d / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
