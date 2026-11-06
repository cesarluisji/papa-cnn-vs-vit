# Imágenes de prueba

70 fotos del **conjunto de prueba** (10 por clase), para probar la app https://agrotec-papa.vercel.app o la demo de Gradio. El modelo **nunca las vio** durante el entrenamiento ni al elegir hiperparámetros.

- **Selección:** se tomaron al azar con `random_state=0` de `results/papa_vit_small_pre_s2/test_predictions.csv`. En *nematodos* el test tiene solo 10 fotos, así que están todas.
- **Formato:** cada carpeta es la clase real. Las imágenes son las versiones preprocesadas de 256×256 que se usaron en la evaluación.
- **`etiquetas.csv`:** tiene la clase real, la predicción y la confianza **tal como las muestra la app** (se pasó cada foto por https://agrotec-papa.vercel.app), si acertó y el archivo de origen.
- **Confianza en el navegador:** el navegador redimensiona la foto de forma algo distinta que Python. La clase predicha coincide en las 70 fotos, pero en casos dudosos la confianza puede cambiar varios puntos.

**Resultado del modelo final en estas 70 fotos:** 60 aciertos (86 %), en línea con el 88.3 % que logra en las 453 fotos de prueba.

| Clase | Aciertos | Errores (predijo…) |
|---|---|---|
| Bacteriosis | 10/10 | — |
| Hongos | 9/10 | hongos_03 → plaga |
| Sana | 10/10 | — |
| Nematodos | 7/10 | nematodos_03 y _06 → hongos; nematodos_10 → plaga |
| Plaga de insectos | 10/10 | — |
| Rancha / tizón tardío | 7/10 | rancha_01, _09 y _10 → hongos |
| Virosis | 7/10 | virosis_07 y _08 → sana; virosis_10 → hongos |

Los errores muestran las mismas confusiones que reporta el artículo, sobre todo con *hongos*. Sirven para enseñar en la exposición en qué falla el modelo. `plaga_insectos_08` acierta pero con 52 % de confianza, así que muestra el aviso de **confianza baja**.

**Datos:** Shabrina et al., *Potato Leaf Disease Dataset in Uncontrolled Environment*, Mendeley Data, DOI 10.17632/ptz377bwb8.1, licencia CC BY 4.0.
