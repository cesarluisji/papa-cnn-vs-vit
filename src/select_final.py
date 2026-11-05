"""Elige el MODELO FINAL para el demo usando SOLO validación (macro-F1), nunca test.

Uso:
    python src/select_final.py                 # elige entre todas las corridas preentrenadas
    python src/select_final.py --run results/papa_vit_small_pre_s2   # forzar una corrida

Copia best.pt + config.json a modelo_final/ y escribe modelo_final/meta.json.
Además copia 1 imagen de TEST por clase a demo/ejemplos/ para los ejemplos del demo.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default="modelo_final")
    args = ap.parse_args()

    cands = []
    for f in glob.glob(f"{args.results}/*/metrics.json"):
        d = Path(f).parent
        if d.name.startswith("_") or not (d / "best.pt").exists():
            continue
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        if m.get("model") == "random_forest":
            continue
        cands.append((m["val"]["macro_f1"], d, m))
    if args.run:
        cands = [c for c in cands if c[1] == Path(args.run)]
    if not cands:
        raise SystemExit("No hay corridas con best.pt")
    cands.sort(key=lambda c: c[0], reverse=True)
    print("Candidatos (ordenados por val macro-F1):")
    for v, d, m in cands[:8]:
        print(f"  {d.name:<30} val {v:.4f} | test {m['test']['macro_f1']:.4f}")
    val, d, m = cands[0]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    shutil.copy(d / "best.pt", out / "best.pt")
    shutil.copy(d / "config.json", out / "config.json")
    meta = dict(run=d.name, model=m["model"], pretrained=m["pretrained"], seed=m["seed"],
                criterio="mayor macro-F1 en validación entre todas las corridas con pesos",
                val=m["val"], test=m["test"], test_f1_por_clase=m["test_f1_por_clase"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMODELO FINAL: {d.name} (val macro-F1 {val:.4f}, test macro-F1 {m['test']['macro_f1']:.4f}) -> {out}/")

    # ejemplos para el demo: 1 imagen de test por clase que el modelo final clasifica bien
    pred = pd.read_csv(d / "test_predictions.csv")
    ok = pred[pred.y_true == pred.y_pred]
    ej = Path("demo/ejemplos")
    if ej.exists():
        shutil.rmtree(ej)
    ej.mkdir(parents=True)
    for c, g in ok.groupby("y_true"):
        src = Path(g.sort_values("path").path.iloc[0])
        shutil.copy(src, ej / f"{c}{src.suffix}")
    print(f"Ejemplos: {sorted(p.name for p in ej.iterdir())}")


if __name__ == "__main__":
    main()
