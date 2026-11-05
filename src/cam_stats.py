"""Medidas simples de concentración de los mapas Grad-CAM (R5).

Uso:  python src/cam_stats.py            (lee results/gradcam/cams.npz, generado por src/gradcam.py)

- masa_top10: fracción de la activación total que cae en el 10 % de píxeles más activos
  (1 = mapa muy concentrado; 0.10 = mapa uniforme).
- masa_centro: fracción de la activación en el cuadrado central (50 % del lado = 25 % del área);
  en este dataset la hoja suele estar centrada, y un mapa uniforme daría 0.25.
Salida: results/tables/gradcam_concentracion.csv
"""
import numpy as np
import pandas as pd

NOMBRES = {"cnn_small": "CNN pequeña", "resnet50": "ResNet-50", "vit_small": "ViT-S/16"}

z = np.load("results/gradcam/cams.npz", allow_pickle=True)
filas = []
for m, nombre in NOMBRES.items():
    c = np.clip(z[m].astype(np.float64), 0, None)
    n, h, w = c.shape
    plano = np.sort(c.reshape(n, -1), 1)[:, ::-1]
    total = plano.sum(1) + 1e-12
    top10 = plano[:, : int(0.1 * h * w)].sum(1) / total
    centro = c[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(n, -1).sum(1) / total
    filas.append(dict(Modelo=nombre, imagenes=n,
                      masa_top10=f"{top10.mean():.2f} ± {top10.std(ddof=1):.2f}",
                      masa_centro=f"{centro.mean():.2f} ± {centro.std(ddof=1):.2f}"))
t = pd.DataFrame(filas)
t.to_csv("results/tables/gradcam_concentracion.csv", index=False)
print(t.to_string(index=False))

# Figura 4 del artículo: 4 filas de results/gradcam/errores.png (mismas imágenes para los 3 modelos)
from pathlib import Path
from PIL import Image
im = Image.open("results/gradcam/errores.png").convert("RGB")
paso, y0 = 233 * 1.38 * im.height / 2760, 125 * 1.38 * im.height / 2760   # filas de la figura de 8 errores
recortes = [im.crop((0, int(y0 + k * paso), im.width, int(y0 + (k + 1) * paso))) for k in (1, 2, 4, 5)]
fig = Image.new("RGB", (im.width, sum(r.height for r in recortes)), "white")
y = 0
for r in recortes:
    fig.paste(r, (0, y)); y += r.height
Path("paper/figs").mkdir(parents=True, exist_ok=True)
fig.resize((fig.width * 2 // 3, fig.height * 2 // 3), Image.LANCZOS).save("paper/figs/gradcam_errores.jpg", quality=88)
print("Figura: paper/figs/gradcam_errores.jpg")
