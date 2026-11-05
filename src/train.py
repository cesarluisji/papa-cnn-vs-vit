"""Entrenamiento + evaluación de un modelo en el dataset de papa.

Uso:
    python src/train.py --config configs/resnet50.yaml --seed 0
    python src/train.py --config configs/resnet50.yaml --seed 0 --pretrained false   # ablación

Selección del mejor checkpoint SOLO con validación (macro-F1). El test se evalúa una única vez
al final, con el mejor checkpoint (checklist pág. 5: test no usado para elegir hiperparámetros).

Salida en results/<run_name>/: config.json, history.csv, metrics.json, test_predictions.csv,
confusion_test.png, best.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, classification_report)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from models import CLASES, build_model, count_params

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULTS = dict(
    model="resnet50", pretrained=True, cultivo="papa",
    split_csv="data/splits/papa_split.csv",
    img_size=224, batch_size=32, epochs=25, lr=1e-4, weight_decay=0.05,
    warmup_epochs=2, patience=7, label_smoothing=0.0, class_weights=True,
    num_workers=4, amp=True, out_dir="results",
)


def str2bool(v):
    return str(v).lower() in {"1", "true", "yes", "si", "sí"}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # rendimiento; la variación entre semillas se reporta como DE


class CSVDataset(Dataset):
    def __init__(self, df, clases, tf):
        self.paths = df["path"].tolist()
        self.y = [clases.index(l) for l in df["label"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.y[i], i


def transforms(img_size):
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    evaluacion = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evaluacion


@torch.no_grad()
def predecir(model, loader, device, amp):
    model.eval()
    ys, ps, probs, idx = [], [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        p = out.float().softmax(1).cpu()
        ys.append(y), ps.append(p.argmax(1)), probs.append(p), idx.append(i)
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy(), torch.cat(idx).numpy())


def metricas(y, p, n):
    return dict(
        macro_f1=f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0),
        balanced_acc=balanced_accuracy_score(y, p),
        accuracy=accuracy_score(y, p),
    )


def evaluar_y_guardar(model, ds, cfg, out, run, hist, t_train, device, amp, seed):
    """Evalúa el mejor checkpoint (ya cargado) en val y TEST y escribe metrics.json, predicciones y confusión."""
    clases = CLASES[cfg["cultivo"]]
    n = len(clases)
    args = argparse.Namespace(seed=seed)
    dl_va = DataLoader(ds["val"], cfg["batch_size"] * 2, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds["test"], cfg["batch_size"] * 2, shuffle=False, num_workers=0)
    yv, pv, _, _ = predecir(model, dl_va, device, amp)
    yt, pt, prob, idx = predecir(model, dl_te, device, amp)
    res = dict(
        run=run, model=cfg["model"], pretrained=cfg["pretrained"], seed=args.seed,
        best_epoch=int(pd.DataFrame(hist)["val_macro_f1"].idxmax() + 1), epochs_run=len(hist),
        val=metricas(yv, pv, n), test=metricas(yt, pt, n),
        test_f1_por_clase=dict(zip(clases, f1_score(yt, pt, average=None, labels=list(range(n)),
                                                     zero_division=0).tolist())),
        train_time_min=t_train / 60,
        gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
        params=cfg["params"],
    )
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    te_df = ds["test"]
    pd.DataFrame(dict(path=[te_df.paths[i] for i in idx], y_true=[clases[i] for i in yt],
                      y_pred=[clases[i] for i in pt], **{f"p_{c}": prob[:, k] for k, c in enumerate(clases)})
                 ).to_csv(out / "test_predictions.csv", index=False)
    cm = confusion_matrix(yt, pt, labels=list(range(n)))
    pd.DataFrame(cm, index=clases, columns=clases).to_csv(out / "confusion_test.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n), clases, rotation=45, ha="right"), ax.set_yticks(range(n), clases)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("Predicho"), ax.set_ylabel("Real"), ax.set_title(run, fontsize=9)
        fig.tight_layout(), fig.savefig(out / "confusion_test.png", dpi=150), plt.close(fig)
    except Exception as e:  # la figura no es crítica
        print("[aviso] sin figura:", e)

    print(classification_report(yt, pt, labels=list(range(n)), target_names=clases, digits=3, zero_division=0))
    print(f"TEST macro-F1 {res['test']['macro_f1']:.4f} | bacc {res['test']['balanced_acc']:.4f} | "
          f"acc {res['test']['accuracy']:.4f} | {res['train_time_min']:.1f} min | "
          f"VRAM pico {res['gpu_peak_mb']:.0f} MB")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained", default=None, help="sobrescribe el config (ablación)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_batches", type=int, default=None, help="solo para prueba rápida")
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR",
                    help="sobrescribe claves del config, p.ej. --set num_workers=2 out_dir=/content/res")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {})
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if args.pretrained is not None:
        cfg["pretrained"] = str2bool(args.pretrained)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["seed"] = args.seed
    if not cfg["pretrained"] and cfg.get("lr_scratch"):
        cfg["lr"] = cfg["lr_scratch"]   # desde cero necesita lr mayor

    pre = "pre" if cfg["pretrained"] else "scratch"
    run = f"{cfg['cultivo']}_{cfg['model']}_{pre}_s{args.seed}{('_' + args.tag) if args.tag else ''}"
    out = Path(cfg["out_dir"]) / run
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg["amp"] and device == "cuda"
    clases = CLASES[cfg["cultivo"]]
    n = len(clases)

    df = pd.read_csv(cfg["split_csv"])
    tf_tr, tf_ev = transforms(cfg["img_size"])
    ds = {s: CSVDataset(df[df.split == s].reset_index(drop=True), clases, tf_tr if s == "train" else tf_ev)
          for s in ["train", "val", "test"]}
    g = torch.Generator().manual_seed(args.seed)
    kw = dict(num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    dl_tr = DataLoader(ds["train"], cfg["batch_size"], shuffle=True, drop_last=True, generator=g, **kw)
    dl_va = DataLoader(ds["val"], cfg["batch_size"] * 2, shuffle=False, **kw)
    dl_te = DataLoader(ds["test"], cfg["batch_size"] * 2, shuffle=False, **kw)

    # pérdida ponderada w_k ∝ 1/n_k con frecuencias de TRAIN (normalizada a media 1)
    cuentas = np.bincount(ds["train"].y, minlength=n)
    w = torch.tensor(1.0 / cuentas, dtype=torch.float32)
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=w.to(device) if cfg["class_weights"] else None,
                               label_smoothing=cfg["label_smoothing"])

    model = build_model(cfg["model"], n, pretrained=cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pasos_ep = len(dl_tr) if args.max_batches is None else min(len(dl_tr), args.max_batches)
    total, warm = cfg["epochs"] * pasos_ep, cfg["warmup_epochs"] * pasos_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=amp)

    cfg.update(run=run, device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               params=count_params(model), n_train=len(ds["train"]), n_val=len(ds["val"]),
               n_test=len(ds["test"]), class_weights_values=w.tolist())
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"== {run} | {cfg['params']/1e6:.2f} M params | train {len(ds['train'])} "
          f"val {len(ds['val'])} test {len(ds['test'])} | {cfg['device']}")

    hist, mejor, sin_mejora = [], -1.0, 0
    t0 = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(cfg["epochs"]):
        model.train()
        te, perdida, vistos = time.perf_counter(), 0.0, 0
        for b, (x, y, _) in enumerate(dl_tr):
            if args.max_batches is not None and b >= args.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            perdida += loss.item() * len(y)
            vistos += len(y)
        yv, pv, _, _ = predecir(model, dl_va, device, amp)
        mv = metricas(yv, pv, n)
        fila = dict(epoch=ep + 1, train_loss=perdida / max(1, vistos), lr=opt.param_groups[0]["lr"],
                    epoch_s=time.perf_counter() - te, **{f"val_{k}": v for k, v in mv.items()})
        hist.append(fila)
        if not np.isfinite(fila["train_loss"]):
            pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
            print(f"ERROR: pérdida no finita en la época {ep+1}; se cancela la corrida", flush=True)
            raise SystemExit(2)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        marca = ""
        if mv["macro_f1"] > mejor:
            mejor, sin_mejora, marca = mv["macro_f1"], 0, " *"
            torch.save(model.state_dict(), out / "best.pt")
        else:
            sin_mejora += 1
        print(f"ep {ep+1:>2} loss {fila['train_loss']:.4f} val F1 {mv['macro_f1']:.4f} "
              f"bacc {mv['balanced_acc']:.4f} acc {mv['accuracy']:.4f} ({fila['epoch_s']:.0f}s){marca}",
              flush=True)
        if sin_mejora >= cfg["patience"]:
            print(f"Parada temprana en época {ep+1}")
            break
    t_train = time.perf_counter() - t0

    # ---- evaluación final en TEST con el mejor checkpoint (según validación) ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    # Liberar los workers persistentes y evaluar sin procesos nuevos: con poca RAM en Windows,
    # crear los workers de test al final colgó la corrida (ResNet s0, 05-nov).
    del dl_tr, dl_va, dl_te
    evaluar_y_guardar(model, ds, cfg, out, run, hist, t_train, device, amp, args.seed)

if __name__ == "__main__":
    main()
