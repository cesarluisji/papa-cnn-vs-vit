"""Prepara el dataset de papa: redimensiona, detecta casi-duplicados y genera la partición sin fuga.

Uso:
    python src/prepare_data.py --raw data/raw/papa --out data/processed/papa_256 --splits data/splits

Pasos
1. Recorre data/raw/papa/<Clase>/*.jpg (7 clases).
2. Redimensiona a 256x256 (las originales son 1500x1500) y guarda en data/processed/papa_256/<clase>/.
3. Calcula un hash perceptual (dHash 64 bits) y agrupa imágenes casi idénticas
   (distancia de Hamming <= --hamming). Las imágenes de un mismo grupo van siempre
   a la misma partición -> evita fuga de datos (R4).
4. Partición estratificada 70/15/15 por grupos con semilla fija -> data/splits/papa_split.csv.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# nombre de carpeta en Mendeley -> nombre de clase en el código (models.CLASES["papa"])
CARPETA_A_CLASE = {
    "bacteria": "bacteria",
    "fungi": "fungi",
    "healthy": "healthy",
    "nematode": "nematode",
    "pest": "pest",
    "phytopthora": "phytophthora",   # así está escrito en Mendeley
    "phytophthora": "phytophthora",
    "virus": "virus",
}
EXT = {".jpg", ".jpeg", ".png"}


def dhash(img: Image.Image, size: int = 8) -> int:
    """Hash de diferencias: 64 bits que cambian poco ante recompresión/redimensionado."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def grupos_por_hash(hashes: np.ndarray, umbral: int) -> np.ndarray:
    """Union-find sobre pares con distancia de Hamming <= umbral."""
    n = len(hashes)
    padre = np.arange(n)

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    h = hashes.astype(np.uint64)
    for i in range(n - 1):
        x = np.bitwise_xor(h[i], h[i + 1:])
        # popcount de 64 bits
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= umbral)[0]:
            a, b = raiz(i), raiz(i + 1 + j)
            if a != b:
                padre[b] = a
    return np.array([raiz(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/papa")
    ap.add_argument("--out", default="data/processed/papa_256")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--hamming", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out, sp = Path(args.raw), Path(args.out), Path(args.splits)
    filas = []
    for carpeta in sorted(p for p in raw.iterdir() if p.is_dir()):
        clase = CARPETA_A_CLASE.get(carpeta.name.lower())
        if clase is None:
            print(f"[aviso] carpeta ignorada: {carpeta}")
            continue
        (out / clase).mkdir(parents=True, exist_ok=True)
        for f in sorted(carpeta.rglob("*")):
            if f.suffix.lower() not in EXT:
                continue
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            destino = out / clase / f.name
            with Image.open(f) as im:
                w, h = im.size
                if destino.exists():          # ya redimensionada en una corrida anterior
                    small = Image.open(destino).convert("RGB")
                else:
                    small = im.convert("RGB").resize((args.size, args.size), Image.LANCZOS)
                    small.save(destino, quality=95)
            filas.append(dict(path=destino.as_posix(), original=f.as_posix(), label=clase,
                              w=w, h=h, sha256=sha, dhash=dhash(small)))
        print(f"{clase:<13} {sum(r['label'] == clase for r in filas):>5}")

    df = pd.DataFrame(filas)
    print(f"Total: {len(df)} imágenes")
    print("Tamaños originales:", df.groupby(["w", "h"]).size().to_dict())

    # --- duplicados exactos ---
    exactos = df[df.duplicated("sha256", keep=False)]
    n_lab = df.groupby("sha256")["label"].transform("nunique")
    conflicto_exacto = df[n_lab > 1].assign(motivo="misma imagen en 2 clases (etiqueta contradictoria)")
    repetida = df[(n_lab == 1) & df.duplicated("sha256", keep="first")].assign(motivo="copia exacta repetida")
    eliminadas = pd.concat([conflicto_exacto, repetida])
    print(f"Duplicados exactos (sha256): {len(exactos)} imágenes en {exactos.sha256.nunique()} pares")
    print(f"  eliminadas por etiqueta contradictoria: {len(conflicto_exacto)}")
    print(f"  eliminadas por copia repetida (se conserva 1): {len(repetida)}")
    sp.mkdir(parents=True, exist_ok=True)
    eliminadas.drop(columns=["dhash"]).to_csv(sp / "papa_eliminadas.csv", index=False)
    df = df.drop(index=eliminadas.index).reset_index(drop=True)
    print(f"Quedan: {len(df)} imágenes")

    # --- casi-duplicados (no exactos): se agrupan para que caigan en la misma partición ---
    df["group"] = grupos_por_hash(df["dhash"].to_numpy(dtype=np.uint64), args.hamming)
    tam = df.groupby("group")["path"].transform("size")
    conflicto = df.groupby("group")["label"].transform("nunique") > 1
    print(f"Imágenes en grupos de casi-duplicados (Hamming<={args.hamming}): {(tam > 1).sum()} "
          f"en {df.loc[tam > 1, 'group'].nunique()} grupos")
    print(f"Grupos con etiquetas distintas (conflicto): {df.loc[conflicto, 'group'].nunique()}")

    # --- partición estratificada por grupo 70/15/15 ---
    g = df.groupby("group").agg(label=("label", lambda s: s.mode()[0]), n=("path", "size")).reset_index()
    g_trval, g_test = train_test_split(g, test_size=0.15, stratify=g["label"], random_state=args.seed)
    g_train, g_val = train_test_split(g_trval, test_size=0.15 / 0.85, stratify=g_trval["label"],
                                      random_state=args.seed)
    asignacion = {**{k: "train" for k in g_train.group}, **{k: "val" for k in g_val.group},
                  **{k: "test" for k in g_test.group}}
    df["split"] = df["group"].map(asignacion)

    sp.mkdir(parents=True, exist_ok=True)
    df["dhash"] = df["dhash"].map(lambda x: f"{x:016x}")
    df.to_csv(sp / "papa_split.csv", index=False)
    tabla = pd.crosstab(df["label"], df["split"], margins=True)[["train", "val", "test", "All"]]
    print(tabla)
    resumen = dict(total_original=len(df) + len(eliminadas), total=len(df), hamming=args.hamming,
                   seed=args.seed, duplicados_exactos=int(len(exactos)),
                   eliminadas_conflicto=int(len(conflicto_exacto)), eliminadas_repetidas=int(len(repetida)),
                   en_grupos_casi_duplicados=int((tam > 1).sum()),
                   grupos_conflicto=int(df.loc[conflicto, "group"].nunique()),
                   por_clase_y_split=tabla.to_dict())
    (sp / "papa_split_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Guardado: {sp / 'papa_split.csv'}")


if __name__ == "__main__":
    main()
