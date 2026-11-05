"""Descarga el dataset de papa desde Mendeley Data y reconstruye EXACTAMENTE la partición del artículo.

Uso (desde la raíz del proyecto):
    python src/download_data.py

Pasos
1. Descarga el zip oficial (≈750 MB) de Mendeley Data:
   Shabrina et al., "Potato Leaf Disease Dataset in Uncontrolled Environment", DOI 10.17632/ptz377bwb8.1 (CC BY 4.0).
2. Lo descomprime en data/raw/papa/<Clase>/.
3. Ejecuta src/prepare_data.py (redimensiona a 256x256, elimina duplicados, agrupa casi-duplicados y parte 70/15/15)
   escribiendo en data/splits_verificacion/, y comprueba que la partición sea idéntica a data/splits/papa_split.csv
   (la que se usó en todos los experimentos).
"""
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://data.mendeley.com/public-api/zip/ptz377bwb8/download/1"
RAIZ = Path(__file__).resolve().parent.parent
ZIP = RAIZ / "data/raw/papa.zip"
RAW = RAIZ / "data/raw/papa"


def descargar():
    if ZIP.exists() and zipfile.is_zipfile(ZIP):
        print(f"Ya existe {ZIP}, no se descarga de nuevo")
        return
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (proyecto academico MIA-203)"})
    with urllib.request.urlopen(req) as r, open(ZIP.with_suffix(".part"), "wb") as f:
        total, leido = int(r.headers.get("Content-Length", 0)), 0
        while bloque := r.read(1 << 20):
            f.write(bloque)
            leido += len(bloque)
            if total:
                print(f"\r  {leido / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="", flush=True)
    print()
    ZIP.with_suffix(".part").rename(ZIP)


def descomprimir():
    if RAW.exists() and any(RAW.iterdir()):
        print(f"Ya existe {RAW}, no se descomprime de nuevo")
        return
    print(f"Descomprimiendo en {RAW} ...")
    with zipfile.ZipFile(ZIP) as z:
        for m in z.infolist():
            partes = Path(m.filename).parts
            if m.is_dir() or len(partes) < 3:      # <carpeta raíz>/<Clase>/<archivo>
                continue
            destino = RAW / partes[-2] / partes[-1]
            destino.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(destino, "wb") as dst:
                shutil.copyfileobj(src, dst)


def preparar_y_verificar():
    verif = RAIZ / "data/splits_verificacion"
    subprocess.run([sys.executable, str(RAIZ / "src/prepare_data.py"), "--raw", "data/raw/papa",
                    "--out", "data/processed/papa_256", "--splits", str(verif.relative_to(RAIZ))],
                   cwd=RAIZ, check=True)
    nuevo = pd.read_csv(verif / "papa_split.csv")[["path", "label", "split"]].sort_values("path")
    oficial = pd.read_csv(RAIZ / "data/splits/papa_split.csv")[["path", "label", "split"]].sort_values("path")
    if nuevo.reset_index(drop=True).equals(oficial.reset_index(drop=True)):
        print(f"\nOK: la partición reconstruida es idéntica a data/splits/papa_split.csv "
              f"({len(oficial)} imágenes: {oficial.split.value_counts().to_dict()})")
        shutil.rmtree(verif)
    else:
        print("\n[ATENCIÓN] La partición reconstruida difiere de la oficial. Los experimentos usan "
              "data/splits/papa_split.csv; revise data/splits_verificacion/.")


if __name__ == "__main__":
    descargar()
    descomprimir()
    preparar_y_verificar()
