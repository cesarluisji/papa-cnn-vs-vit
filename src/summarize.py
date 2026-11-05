"""Junta todas las corridas de results/ en tablas y figuras para el paper.

Uso:
    python src/summarize.py                # lee results/*/metrics.json
    python src/summarize.py --results results --out results

Salidas
  results/tables/resumen.{csv,md,tex}        media ± DE en TEST por configuración (+ líneas base publicadas)
  results/tables/f1_por_clase.{csv,tex}      F1 por clase (media ± DE)
  results/tables/ablacion.{csv,tex}          preentrenado vs desde cero (Δ macro-F1)
  results/tables/confusiones_top.csv         pares (real -> predicho) más frecuentes, sumando semillas
  results/tables/mcnemar.csv                 ResNet-50 vs ViT por semilla (mismas imágenes de test)
  results/tables/costo_gpu.csv               minutos y VRAM por configuración y equipo (R6)
  results/figures/f1_por_clase.png           barras agrupadas con DE
  results/figures/confusion_<config>.png     matriz normalizada por fila, sumando las semillas
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES

CLASES_P = CLASES["papa"]
# orden fijo de configuraciones (y de colores: paleta categórica validada, slots 1-4)
ORDEN = [("random_forest", False), ("cnn_small", False), ("resnet50", True), ("vit_small", True),
         ("resnet50", False), ("vit_small", False)]
NOMBRE = {("random_forest", False): "Random Forest (color+HOG+LBP)",
          ("cnn_small", False): "CNN pequeña (desde cero)",
          ("resnet50", True): "ResNet-50 (preentrenado)",
          ("vit_small", True): "ViT-S/16 (preentrenado)",
          ("resnet50", False): "ResNet-50 (desde cero)",
          ("vit_small", False): "ViT-S/16 (desde cero)"}
CORTO = {("random_forest", False): "RF", ("cnn_small", False): "CNN pequeña",
         ("resnet50", True): "ResNet-50", ("vit_small", True): "ViT-S/16",
         ("resnet50", False): "ResNet-50 cero", ("vit_small", False): "ViT-S/16 cero"}
COLOR = {("random_forest", False): "#2a78d6", ("cnn_small", False): "#eb6834",
         ("resnet50", True): "#1baf7a", ("vit_small", True): "#eda100"}
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
# Shabrina et al., Data in Brief 52 (2024) 109955, Tabla 6 (sin aumento): partición 90/10, 1 corrida,
# sin deduplicar. Su F1 coincide con un promedio ponderado (recall = accuracy) -> NO es macro-F1.
PUBLICADO = [("EfficientNetV2B3 [Shabrina 2024]", 0.7363, 0.7302),
             ("MobileNetV3-Large [Shabrina 2024]", 0.7203, 0.7131),
             ("ResNet-50 [Shabrina 2024]", 0.6817, 0.6748)]


def ms(v, d=3):
    v = np.asarray(v, dtype=float)
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def cargar(results):
    filas = []
    for f in sorted(glob.glob(f"{results}/*/metrics.json")):
        if Path(f).parent.name.startswith("_"):
            continue  # corridas descartadas
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        cfgp = Path(f).parent / "config.json"
        dev = json.loads(cfgp.read_text(encoding="utf-8")).get("device", "cpu") if cfgp.exists() else "cpu"
        filas.append(dict(dir=str(Path(f).parent), key=(m["model"], bool(m["pretrained"])), seed=m["seed"],
                          device=dev, **{f"test_{k}": v for k, v in m["test"].items()},
                          **{f"val_{k}": v for k, v in m["val"].items()},
                          **{f"f1_{c}": m["test_f1_por_clase"][c] for c in CLASES_P},
                          minutos=m["train_time_min"], vram_mb=m.get("gpu_peak_mb", 0),
                          best_epoch=m.get("best_epoch", np.nan), params=m.get("params", np.nan)))
    return pd.DataFrame(filas)


def a_latex(df, caption, label):
    cols = "l" + "c" * (df.shape[1] - 1)
    lineas = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
              r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{" + cols + "}", r"\toprule",
              " & ".join(df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lineas.append(" & ".join(str(x).replace("±", r"$\pm$").replace("_", r"\_") for x in r) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    tab, fig = Path(args.out) / "tables", Path(args.out) / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    df = cargar(args.results)
    if df.empty:
        print("No hay corridas en", args.results)
        return
    keys = [k for k in ORDEN if k in set(df.key)]

    # ---------- 1. resumen ----------
    filas = []
    for k in keys:
        d = df[df.key == k]
        filas.append({"Modelo": NOMBRE[k], "Semillas": len(d),
                      "Macro-F1": ms(d.test_macro_f1), "Balanced acc.": ms(d.test_balanced_acc),
                      "Accuracy": ms(d.test_accuracy), "Val macro-F1": ms(d.val_macro_f1),
                      "Params (M)": f"{d.params.iloc[0] / 1e6:.1f}" if d.params.notna().any() else "—"})
    for nombre, acc, f1w in PUBLICADO:
        filas.append({"Modelo": nombre, "Semillas": 1, "Macro-F1": f"({f1w:.3f})†",
                      "Balanced acc.": "—", "Accuracy": f"{acc:.3f}", "Val macro-F1": "—", "Params (M)": "—"})
    res = pd.DataFrame(filas)
    res.to_csv(tab / "resumen.csv", index=False)
    nota = ("† F1 ponderado reportado por Shabrina et al. (2024), partición 90/10 sin deduplicar y 1 corrida; "
            "no es directamente comparable.")
    (tab / "resumen.md").write_text(res.to_markdown(index=False) + "\n\n" + nota + "\n", encoding="utf-8")
    (tab / "resumen.tex").write_text(a_latex(res.drop(columns=["Val macro-F1"]),
                                             "Resultados en el conjunto de prueba (media $\\pm$ DE sobre semillas). " + nota,
                                             "tab:resumen"), encoding="utf-8")
    print(res.to_string(index=False))

    # ---------- 2. F1 por clase ----------
    f1c = pd.DataFrame({CORTO[k]: [ms(df[df.key == k][f"f1_{c}"], 2) for c in CLASES_P] for k in keys},
                       index=CLASES_P)
    f1c.index.name = "Clase"
    f1c.to_csv(tab / "f1_por_clase.csv")
    (tab / "f1_por_clase.tex").write_text(a_latex(f1c.reset_index(), "F1 por clase en prueba (media $\\pm$ DE).",
                                                  "tab:f1clase"), encoding="utf-8")

    # ---------- 3. ablación ----------
    abl = []
    for m in ["resnet50", "vit_small"]:
        a, b = df[df.key == (m, True)], df[df.key == (m, False)]
        if len(a) and len(b):
            abl.append({"Modelo": CORTO[(m, True)], "Preentrenado": ms(a.test_macro_f1),
                        "Desde cero": ms(b.test_macro_f1),
                        "Δ macro-F1": f"{a.test_macro_f1.mean() - b.test_macro_f1.mean():+.3f}",
                        "Semillas (pre/cero)": f"{len(a)}/{len(b)}"})
    if abl:
        abl = pd.DataFrame(abl)
        abl.to_csv(tab / "ablacion.csv", index=False)
        (tab / "ablacion.tex").write_text(a_latex(abl, "Ablación: preentrenamiento en ImageNet vs. entrenamiento "
                                                       "desde cero (macro-F1 en prueba).", "tab:ablacion"),
                                          encoding="utf-8")
        print("\nAblación\n", abl.to_string(index=False))

    # ---------- 4. confusiones (sumando semillas) ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    azul = LinearSegmentedColormap.from_list("azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab",
                                                      "#104281"])
    top = []
    for k in keys:
        cm = np.zeros((7, 7), dtype=int)
        for d in df[df.key == k].dir:
            p = pd.read_csv(Path(d) / "test_predictions.csv")
            for t, q in zip(p.y_true, p.y_pred):
                cm[CLASES_P.index(t), CLASES_P.index(q)] += 1
        n_sem = int((df.key == k).sum())
        for i in range(7):
            for j in range(7):
                if i != j and cm[i, j]:
                    top.append({"Modelo": CORTO[k], "Real": CLASES_P[i], "Predicho": CLASES_P[j],
                                "Errores (suma semillas)": int(cm[i, j]), "Semillas": n_sem,
                                "% de la clase real": round(100 * cm[i, j] / cm[i].sum(), 1)})
        cmn = cm / cm.sum(1, keepdims=True)
        f, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cmn, cmap=azul, vmin=0, vmax=1)
        ax.set_xticks(range(7), CLASES_P, rotation=45, ha="right")
        ax.set_yticks(range(7), CLASES_P)
        for i in range(7):
            for j in range(7):
                if cmn[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if cmn[i, j] > 0.5 else INK)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"{NOMBRE[k]}\n% por fila, {n_sem} semilla(s)", fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        f.tight_layout()
        f.savefig(fig / f"confusion_{k[0]}_{'pre' if k[1] else 'cero'}.png", dpi=200, facecolor=SURF)
        plt.close(f)
    top = pd.DataFrame(top).sort_values(["Modelo", "Errores (suma semillas)"], ascending=[True, False])
    top.to_csv(tab / "confusiones_top.csv", index=False)
    print("\nConfusiones más frecuentes (top 3 por modelo)\n",
          top.groupby("Modelo").head(3).to_string(index=False))

    # ---------- 5. McNemar ResNet-50 vs ViT (misma semilla, mismas imágenes) ----------
    mc = []
    for s in sorted(set(df.seed)):
        a = df[(df.key == ("resnet50", True)) & (df.seed == s)]
        b = df[(df.key == ("vit_small", True)) & (df.seed == s)]
        if len(a) and len(b):
            pa = pd.read_csv(Path(a.dir.iloc[0]) / "test_predictions.csv").set_index("path")
            pb = pd.read_csv(Path(b.dir.iloc[0]) / "test_predictions.csv").set_index("path").loc[pa.index]
            ca, cb = pa.y_pred == pa.y_true, pb.y_pred == pb.y_true
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            mc.append({"Semilla": s, "Solo ResNet acierta": n01, "Solo ViT acierta": n10, "p (McNemar exacto)": round(p, 4)})
    if mc:
        pd.DataFrame(mc).to_csv(tab / "mcnemar.csv", index=False)
        print("\nMcNemar ResNet-50 vs ViT\n", pd.DataFrame(mc).to_string(index=False))

    # ---------- 6. costo (R6) ----------
    costo = df.groupby([df.key.map(NOMBRE), "device"]).agg(
        corridas=("seed", "size"), min_por_corrida=("minutos", "mean"), min_total=("minutos", "sum"),
        vram_pico_mb=("vram_mb", "max")).round(1).reset_index().rename(columns={"key": "Modelo"})
    costo.to_csv(tab / "costo_gpu.csv", index=False)
    gpu = df[df.device != "cpu"]
    print(f"\nCosto: {df.minutos.sum() / 60:.2f} h en total; {gpu.minutos.sum() / 60:.2f} h de GPU "
          f"({', '.join(sorted(set(gpu.device)))})")

    # ---------- 7. figura F1 por clase (configuraciones principales) ----------
    princ = [k for k in keys if k in COLOR]
    x = np.arange(len(CLASES_P))
    w = 0.8 / len(princ)
    f, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, k in enumerate(princ):
        d = df[df.key == k]
        mu = np.array([d[f"f1_{c}"].mean() for c in CLASES_P])
        sd = np.array([d[f"f1_{c}"].std(ddof=1) if len(d) > 1 else 0 for c in CLASES_P])
        ax.bar(x + (i - (len(princ) - 1) / 2) * w, mu, w * 0.9, yerr=sd, color=COLOR[k], label=NOMBRE[k],
               edgecolor=SURF, linewidth=1, error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=1.5))
    soporte = pd.read_csv(Path(df.dir.iloc[0]) / "test_predictions.csv").y_true.value_counts()
    ax.set_xticks(x, [f"{c}\n(n={soporte.get(c, 0)})" for c in CLASES_P])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 en prueba")
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7)
    f.tight_layout()
    f.savefig(fig / "f1_por_clase.png", dpi=200, facecolor=SURF)
    plt.close(f)
    print(f"\nTablas en {tab}/ y figuras en {fig}/")


if __name__ == "__main__":
    main()
