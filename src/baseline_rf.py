"""Línea base clásica (R3): descriptores de color + textura + Random Forest.

Uso:
    python src/baseline_rf.py --seeds 0 1 2

Descriptores por imagen (de la versión 256x256):
- Histograma HSV (16 bins por canal, normalizado)            -> 48
- Media y desviación estándar por canal RGB y HSV             -> 12
- HOG en escala de grises 128x128 (9 orientaciones, celdas 16) -> 1764
- LBP uniforme (P=8, R=1), histograma de 10 bins              -> 10

Hiperparámetros elegidos SOLO con validación (macro-F1); test se evalúa una vez al final.
Salida: results/papa_rf_s<seed>/metrics.json (mismo formato que train.py) y test_predictions.csv.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES


def descriptores(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    hist = np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 1))[0] for c in range(3)]).astype(np.float32)
    hist /= hist.sum() / 3
    stats = np.concatenate([rgb.reshape(-1, 3).mean(0), rgb.reshape(-1, 3).std(0),
                            hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0)])
    gris = rgb2gray(rgb)
    g128 = np.asarray(Image.fromarray((gris * 255).astype(np.uint8)).resize((128, 128)), dtype=np.float32) / 255
    h = hog(g128, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern((gris * 255).astype(np.uint8), P=8, R=1, method="uniform")
    hl = np.histogram(lbp, bins=10, range=(0, 10))[0].astype(np.float32)
    hl /= hl.sum()
    return np.concatenate([hist, stats, h, hl]).astype(np.float32)


def metricas(y, p, n):
    return dict(macro_f1=f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0),
                balanced_acc=balanced_accuracy_score(y, p), accuracy=accuracy_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", default="data/splits/papa_split.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    clases = CLASES["papa"]
    n = len(clases)
    df = pd.read_csv(args.split_csv)
    cache = Path(args.out_dir) / "rf_features.npz"
    t0 = time.perf_counter()
    if cache.exists():
        X = np.load(cache)["X"]
    else:
        X = np.stack(Parallel(n_jobs=args.n_jobs)(delayed(descriptores)(p) for p in df.path))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X)
    t_feat = time.perf_counter() - t0
    y = df.label.map(clases.index).to_numpy()
    tr, va, te = (df.split == s for s in ["train", "val", "test"])
    print(f"Descriptores: {X.shape} en {t_feat/60:.1f} min")

    grid = dict(n_estimators=[300, 600], max_features=["sqrt", 0.05], min_samples_leaf=[1, 3])
    for seed in args.seeds:
        t1 = time.perf_counter()
        mejor, mejor_hp = -1, None
        for vals in itertools.product(*grid.values()):
            hp = dict(zip(grid, vals))
            rf = RandomForestClassifier(**hp, class_weight="balanced", n_jobs=args.n_jobs, random_state=seed)
            rf.fit(X[tr], y[tr])
            f1 = metricas(y[va], rf.predict(X[va]), n)["macro_f1"]
            print(f"  seed {seed} {hp} val macro-F1 {f1:.4f}")
            if f1 > mejor:
                mejor, mejor_hp = f1, hp
        rf = RandomForestClassifier(**mejor_hp, class_weight="balanced", n_jobs=args.n_jobs, random_state=seed)
        rf.fit(X[tr], y[tr])
        pv, pt = rf.predict(X[va]), rf.predict(X[te])
        prob = rf.predict_proba(X[te])
        run = f"papa_rf_s{seed}"
        out = Path(args.out_dir) / run
        out.mkdir(parents=True, exist_ok=True)
        res = dict(run=run, model="random_forest", pretrained=False, seed=seed, hiperparametros=mejor_hp,
                   n_features=int(X.shape[1]), val=metricas(y[va], pv, n), test=metricas(y[te], pt, n),
                   test_f1_por_clase=dict(zip(clases, f1_score(y[te], pt, average=None, labels=list(range(n)),
                                                                zero_division=0).tolist())),
                   train_time_min=(time.perf_counter() - t1) / 60, feature_time_min=t_feat / 60, gpu_peak_mb=0)
        (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        pd.DataFrame(dict(path=df.path[te].values, y_true=[clases[i] for i in y[te]], y_pred=[clases[i] for i in pt],
                          **{f"p_{c}": prob[:, k] for k, c in enumerate(clases)})).to_csv(out / "test_predictions.csv", index=False)
        pd.DataFrame(confusion_matrix(y[te], pt, labels=list(range(n))), index=clases, columns=clases
                     ).to_csv(out / "confusion_test.csv")
        print(classification_report(y[te], pt, labels=list(range(n)), target_names=clases, digits=3, zero_division=0))
        print(f"{run}: mejor {mejor_hp} | TEST macro-F1 {res['test']['macro_f1']:.4f} "
              f"bacc {res['test']['balanced_acc']:.4f} acc {res['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
