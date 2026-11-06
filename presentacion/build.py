"""Exporta presentacion/presentacion.html a PDF con Chrome sin interfaz.

    python presentacion/build.py            # -> presentacion/presentacion.pdf

Las figuras de presentacion/img/ salen de results/figures/, paper/figs/ y demo/ejemplos/
(se copian con --figuras). Los números de las diapositivas son los de results/tables/.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAIZ = HERE.parent
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "google-chrome", "chromium"]
FIGURAS = {"results/figures/dataset.png": "dataset.png",
           "results/figures/curvas_entrenamiento.png": "curvas_entrenamiento.png",
           "results/figures/f1_por_clase.png": "f1_por_clase.png",
           "results/figures/confusion_vit_small_pre.png": "confusion_vit_small_pre.png",
           "paper/figs/gradcam_errores.jpg": "gradcam_errores.jpg",
           "paper/figs/app_web.jpg": "app_web.jpg"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figuras", action="store_true", help="vuelve a copiar las figuras del proyecto")
    p.add_argument("--out", default=str(HERE / "presentacion.pdf"))
    a = p.parse_args()
    if a.figuras:
        for src, dst in FIGURAS.items():
            shutil.copy(RAIZ / src, HERE / "img" / dst)
        for c in ["bacteria", "fungi", "healthy", "nematode", "pest", "phytophthora", "virus"]:
            shutil.copy(RAIZ / f"demo/ejemplos/{c}.jpg", HERE / "img" / f"ej_{c}.jpg")
    chrome = next((c for c in CHROME if Path(c).exists() or shutil.which(c)), None)
    if chrome is None:
        raise SystemExit("No se encontró Chrome/Edge: abra presentacion.html e imprima a PDF.")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={a.out}",
                    (HERE / "presentacion.html").as_uri()], check=True, capture_output=True)
    print("PDF:", a.out)


if __name__ == "__main__":
    main()
