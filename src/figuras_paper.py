"""Figuras adicionales del artículo: conjunto de datos y curvas de entrenamiento.

Uso (desde la raíz):
    python src/figuras_paper.py
Escribe results/figures/{dataset,curvas_entrenamiento}.png y las copia a paper/figs/.
No necesita GPU ni el dataset completo: usa data/splits/papa_split.csv, demo/ejemplos/ y los history.csv.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES

RAIZ = Path(__file__).resolve().parent.parent
FIG, PAPER = RAIZ / "results/figures", RAIZ / "paper/figs"
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e1"
ES = {"bacteria": "Bacteria", "fungi": "Hongos", "healthy": "Sana", "nematode": "Nematodos",
      "pest": "Plaga", "phytophthora": "Phytophthora", "virus": "Virus"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def guardar(f, nombre):
    f.savefig(FIG / nombre, dpi=220, facecolor=SURF, bbox_inches="tight", pad_inches=0.03)
    shutil.copy(FIG / nombre, PAPER / nombre)
    plt.close(f)
    print("  ", FIG / nombre)


def figura_dataset():
    clases = CLASES["papa"]
    df = pd.read_csv(RAIZ / "data/splits/papa_split.csv")
    cnt = df.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(clases)[["train", "val", "test"]]

    f = plt.figure(figsize=(7.16, 2.75), facecolor=SURF)
    gs = f.add_gridspec(2, 7, height_ratios=[1.0, 0.95], hspace=0.30, wspace=0.06)
    for i, c in enumerate(clases):
        ax = f.add_subplot(gs[0, i])
        ax.imshow(Image.open(RAIZ / f"demo/ejemplos/{c}.jpg").convert("RGB"))
        ax.set_title(ES[c], fontsize=8, style="italic" if c == "phytophthora" else "normal", pad=3)
        ax.axis("off")

    ax = f.add_subplot(gs[1, :])
    y = np.arange(len(clases))[::-1]
    izq = np.zeros(len(clases))
    for parte, nombre, col in [("train", "Entrenamiento", "#1c5cab"), ("val", "Validación", "#5f9ae6"),
                               ("test", "Prueba", "#a9cbf5")]:
        v = cnt[parte].to_numpy()
        ax.barh(y, v, left=izq, height=0.68, color=col, edgecolor=SURF, linewidth=1, label=nombre)
        izq += v
    for yi, tot in zip(y, izq):
        ax.text(tot + 8, yi, f"{int(tot)}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [ES[c] for c in clases])
    ax.set_xlabel("Número de imágenes (tras eliminar duplicados; total 3 015)")
    ax.set_xlim(0, izq.max() * 1.1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.98), frameon=False, fontsize=7, ncol=3)
    guardar(f, "dataset.png")


CURVAS = [("cnn_small", "scratch", "CNN pequeña (cero)", "#eb6834", "-"),
          ("resnet50", "pre", "ResNet-50 (pre.)", "#1baf7a", "-"),
          ("vit_small", "pre", "ViT-S/16 (pre.)", "#eda100", "-"),
          ("resnet50", "scratch", "ResNet-50 (cero)", "#1baf7a", "--"),
          ("vit_small", "scratch", "ViT-S/16 (cero)", "#eda100", "--")]


def figura_curvas():
    f, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), facecolor=SURF)
    for modelo, pre, nombre, col, ls in CURVAS:
        h = [pd.read_csv(RAIZ / f"results/papa_{modelo}_{pre}_s{s}/history.csv") for s in (0, 1, 2)]
        n = min(len(x) for x in h)
        ep = h[0].epoch.to_numpy()[:n]
        for ax, colname in zip(axs, ["train_loss", "val_macro_f1"]):
            v = np.stack([x[colname].to_numpy()[:n] for x in h])
            mu, sd = v.mean(0), v.std(0, ddof=1)
            ax.plot(ep, mu, color=col, ls=ls, lw=1.6, label=nombre)
            ax.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    axs[0].set_ylabel("Pérdida de entrenamiento\n(entropía cruzada ponderada)")
    axs[1].set_ylabel("Macro-F1 en validación")
    axs[1].set_ylim(0, 1)
    for ax, t in zip(axs, ["(a) Pérdida", "(b) Validación"]):
        ax.set_xlabel("Época")
        ax.set_xlim(1, 40)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(t, fontsize=8, loc="left")
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    f.tight_layout(w_pad=2.5, rect=(0, 0.08, 1, 1))
    guardar(f, "curvas_entrenamiento.png")


if __name__ == "__main__":
    PAPER.mkdir(exist_ok=True)
    figura_dataset()
    figura_curvas()
