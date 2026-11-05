# Enfermedades en hojas de papa: CNN vs Vision Transformer con pocos datos

Proyecto final de **MIA-203 Redes Neuronales y Aprendizaje Profundo** (Maestría en IA, UNI FIIS).

Clasificamos fotos de hojas de papa tomadas en campo en **7 clases**: bacteria, hongos, sana, nematodos, plaga de insectos, *Phytophthora* (rancha) y virus.
- **Pregunta central:** con solo ~2,100 imágenes de entrenamiento, ¿rinde mejor una CNN (ResNet-50) o un Vision Transformer (ViT-S/16)?
- **Pregunta secundaria:** ¿cuánto aporta el preentrenamiento en ImageNet?

**Demo en Colab:** [`notebooks/demo_colab.ipynb`](notebooks/demo_colab.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabo2607/papa-cnn-vs-vit/blob/main/notebooks/demo_colab.ipynb). Descarga los pesos, clasifica fotos y muestra el Grad-CAM. No necesita el dataset ni GPU.

## Resultados principales

Conjunto de **prueba** (453 imágenes nunca usadas para entrenar ni para elegir hiperparámetros). Media ± desviación estándar sobre **3 semillas**.

| Modelo | Macro-F1 | Balanced acc. | Accuracy | Params |
|---|---|---|---|---|
| Random Forest (color + HOG + LBP) | 0.464 ± 0.021 | 0.452 ± 0.014 | 0.558 ± 0.010 | — |
| CNN pequeña (desde cero) | 0.614 ± 0.011 | 0.661 ± 0.004 | 0.630 ± 0.009 | 1.2 M |
| ResNet-50 (desde cero) | 0.642 ± 0.007 | 0.687 ± 0.013 | 0.659 ± 0.003 | 23.5 M |
| ViT-S/16 (desde cero) | 0.499 ± 0.028 | 0.550 ± 0.019 | 0.497 ± 0.029 | 21.7 M |
| **ResNet-50 (preentrenado ImageNet)** | **0.843 ± 0.012** | 0.845 ± 0.011 | 0.855 ± 0.005 | 23.5 M |
| **ViT-S/16 (preentrenado ImageNet)** | **0.853 ± 0.013** | 0.849 ± 0.008 | 0.869 ± 0.016 | 21.7 M |

- **Preentrenado, CNN ≈ ViT.** La prueba de McNemar exacta (ResNet vs ViT, por semilla) da p = 0.38, 1.00 y 0.16: no hay diferencia significativa.
- **El preentrenamiento es decisivo**, sobre todo para el ViT: +0.355 de macro-F1, contra +0.201 en ResNet. Desde cero, el ViT queda incluso por debajo de la CNN pequeña.
- **Error más frecuente:** confundir *plaga* con *hongos*. Le pasa al 14 % de las fotos de plaga en ViT y al 11 % en ResNet.
- **Modelo final** (elegido por macro-F1 de **validación**): ViT-S/16 preentrenado, semilla 2. En prueba logra macro-F1 0.864 y accuracy 0.883.
- **Referencia publicada:** Shabrina et al. (2024) reportan accuracy de 0.682 (ResNet-50) y 0.736 (EfficientNetV2-B3) en este dataset. Usaron otra partición (90/10, sin deduplicar y 1 corrida), así que la comparación es solo orientativa.

Tablas completas en [`results/tables/`](results/tables), figuras en [`results/figures/`](results/figures) y [`results/gradcam/`](results/gradcam).

## Estructura

```
├── README.md
├── requirements.txt
├── data/            README con fuente, licencia y limpieza; splits/ con la partición exacta (las imágenes no se suben)
├── src/
│   ├── download_data.py   descarga de Mendeley + preprocesamiento + verificación de la partición
│   ├── prepare_data.py    256×256, deduplicación (sha256 + dHash), partición 70/15/15 por grupos
│   ├── models.py          CNN pequeña, ResNet-50 y ViT-S/16 (timm)
│   ├── train.py           entrenamiento y evaluación (mejor época por macro-F1 de validación; test una sola vez)
│   ├── eval_run.py        re-evalúa una corrida desde best.pt sin reentrenar
│   ├── baseline_rf.py     línea base Random Forest
│   ├── summarize.py       tablas (CSV/Markdown/LaTeX), McNemar, costo y figuras del artículo
│   ├── gradcam.py         mapas Grad-CAM (aciertos por clase y errores)
│   ├── select_final.py    elige el modelo final por validación → modelo_final/
│   └── demo_app.py        app web (Gradio): foto → enfermedad + confianza + Grad-CAM
├── configs/         hiperparámetros de cada modelo (YAML)
├── notebooks/       demo_colab.ipynb (demo), entrenar_colab.ipynb (reproducir), ejecutados/ (corridas tal como se hicieron)
├── modelo_final/    config.json y meta.json del modelo final (los pesos se descargan del Release)
├── demo/ejemplos/   1 foto de prueba por clase
├── paper/           artículo IEEE (LaTeX)
└── results/         una carpeta por corrida (config, historia, métricas, predicciones en test, matriz de confusión),
                     tables/, figures/, gradcam/, logs/
```

## Cómo reproducir

### 1. Instalación
```bash
git clone https://github.com/gabo2607/papa-cnn-vs-vit.git
cd papa-cnn-vs-vit
pip install -r requirements.txt
# GPU NVIDIA: pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Datos (≈750 MB; ~5–10 min)
```bash
python src/download_data.py
```
Descarga el dataset oficial, lo preprocesa y verifica que la partición sea **idéntica** a `data/splits/papa_split.csv`. Detalles en [`data/README.md`](data/README.md).

### 3. Entrenar y evaluar
Todas las redes usan 40 épocas, AdamW, *warmup* + coseno, entropía cruzada ponderada por clase, imágenes de 224×224 y semillas 0, 1 y 2. Cada corrida guarda en `results/papa_<modelo>_<pre|scratch>_s<semilla>/`.

```bash
# Línea base (CPU, ~2 min por semilla)
python src/baseline_rf.py --seeds 0 1 2

# Redes (≈15 min por corrida en una T4)
for s in 0 1 2; do
  python src/train.py --config configs/cnn_small.yaml --seed $s --pretrained false
  python src/train.py --config configs/resnet50.yaml  --seed $s --pretrained true
  python src/train.py --config configs/vit_small.yaml --seed $s --pretrained true
  python src/train.py --config configs/resnet50.yaml  --seed $s --pretrained false   # ablación
  python src/train.py --config configs/vit_small.yaml --seed $s --pretrained false   # ablación
done
```
En Colab: [`notebooks/entrenar_colab.ipynb`](notebooks/entrenar_colab.ipynb).

### 4. Regenerar tablas, figuras y Grad-CAM del artículo
No hace falta reentrenar: las métricas y predicciones de las 21 corridas ya están en `results/`.
```bash
python src/summarize.py        # results/tables/*  (resumen, ablación, F1 por clase, McNemar, costo, confusiones)
                               # results/figures/* (F1 por clase, matrices de confusión)
python src/gradcam.py --runs results/papa_cnn_small_scratch_s2 results/papa_resnet50_pre_s1 results/papa_vit_small_pre_s2 \
       --n_por_clase 2 --n_errores 8 --out results/gradcam     # requiere los best.pt (reentrenar o descargar)
python src/select_final.py     # elige el modelo final por validación → modelo_final/
```

### 5. Demo
```bash
python src/demo_app.py           # http://127.0.0.1:7860   (descarga los pesos la primera vez)
python src/demo_app.py --share   # además, enlace público temporal
```

## Pesos

El modelo final (ViT-S/16, ≈83 MB) está publicado en el Release [`modelo-final`](https://github.com/gabo2607/papa-cnn-vs-vit/releases/tag/modelo-final). `src/demo_app.py` y el notebook de demo lo descargan solos.

## Notas de cómputo

- **Hardware:** Google Colab (Tesla T4) y una laptop con GTX 1650 Ti de 4 GB. Costo total ≈ 6 h de GPU (`results/tables/costo_gpu.csv`).
- **CNN pequeña sin fp16:** en la GTX 1650 Ti, la precisión mixta (fp16) hacía colapsar el entrenamiento, por eso esta red entrena en fp32 (`amp: false`).
- **ResNet-50 preentrenada, semilla 1:** en la GTX 1650 Ti la validación colapsó a una sola clase, así que se repitió en la T4. La semilla 0 de este modelo es de la laptop; las semillas 1 y 2, de la T4.

## Equipo

Ver la sección *Contribuciones* del artículo.

## Licencia y cita de los datos

- **Código:** MIT.
- **Datos:** CC BY 4.0, de Shabrina et al., *Potato Leaf Disease Dataset in Uncontrolled Environment*, Mendeley Data, DOI 10.17632/ptz377bwb8.1.
