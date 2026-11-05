# Datos

**No se suben imágenes al repositorio.** Todo se descarga y reconstruye con un comando:

```bash
python src/download_data.py
```

Este script:
1. Descarga el zip oficial (≈750 MB) de Mendeley Data.
2. Lo descomprime en `data/raw/papa/<Clase>/`.
3. Ejecuta `src/prepare_data.py`: redimensiona a 256×256, elimina duplicados, agrupa casi-duplicados y parte 70/15/15.
4. **Verifica que la partición reconstruida sea idéntica** a `data/splits/papa_split.csv`, que es la que usan todos los experimentos.

## Fuente y licencia

| | |
|---|---|
| Dataset | *Potato Leaf Disease Dataset in Uncontrolled Environment* |
| Autores | Shabrina, N. H.; Indarti, S.; Maharani, R.; Kristiyanti, D. A.; Irmawati, I.; Prastomo, N.; Adillah M, T. |
| Repositorio | Mendeley Data, V1, DOI [10.17632/ptz377bwb8.1](https://doi.org/10.17632/ptz377bwb8.1) |
| Artículo | *Data in Brief* 52 (2024) 109955, DOI [10.1016/j.dib.2023.109955](https://doi.org/10.1016/j.dib.2023.109955) |
| Licencia | **CC BY 4.0** (permite compartir y modificar citando la fuente). Cambios hechos aquí: redimensionado a 256×256 y eliminación de duplicados |
| Contenido | 3,076 fotos JPG 1500×1500 de hojas de papa en campo (Java Central, Indonesia), 7 clases |

## Limpieza y partición (`data/splits/`)

| Archivo | Contenido |
|---|---|
| `papa_split.csv` | Una fila por imagen usada: ruta, clase, sha256, dHash, grupo de casi-duplicados y partición (`train`/`val`/`test`) |
| `papa_eliminadas.csv` | 61 imágenes eliminadas: 12 por estar **repetidas en 2 clases distintas** (etiqueta contradictoria) y 49 copias exactas repetidas (se conserva 1) |
| `papa_split_resumen.json` | Conteos por clase y partición |

- **Duplicados exactos** (sha256): 55 pares. Se eliminan las copias y las imágenes con etiqueta contradictoria. Quedan 3,076 → **3,015** imágenes.
- **Casi-duplicados** (dHash de 64 bits, distancia de Hamming ≤ 4): se agrupan con *union-find* y cada grupo cae completo en una sola partición. Así se evita la fuga de datos entre entrenamiento y prueba.
- **Partición** estratificada por grupo 70/15/15 con semilla 42:

| Clase | train | val | test | total |
|---|---:|---:|---:|---:|
| bacteria | 397 | 86 | 86 | 569 |
| fungi | 518 | 110 | 112 | 740 |
| healthy | 139 | 30 | 30 | 199 |
| nematode | 47 | 10 | 10 | 67 |
| pest | 421 | 90 | 90 | 601 |
| phytophthora | 217 | 46 | 46 | 309 |
| virus | 373 | 78 | 79 | 530 |
| **total** | **2112** | **450** | **453** | **3015** |
