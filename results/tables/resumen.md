| Modelo                            |   Semillas | Macro-F1      | Balanced acc.   | Accuracy      | Val macro-F1   | Params (M)   |
|:----------------------------------|-----------:|:--------------|:----------------|:--------------|:---------------|:-------------|
| Random Forest (color+HOG+LBP)     |          3 | 0.464 ± 0.021 | 0.452 ± 0.014   | 0.558 ± 0.010 | 0.481 ± 0.009  | —            |
| CNN pequeña (desde cero)          |          3 | 0.614 ± 0.011 | 0.661 ± 0.004   | 0.630 ± 0.009 | 0.624 ± 0.014  | 1.2          |
| ResNet-50 (preentrenado)          |          3 | 0.843 ± 0.012 | 0.845 ± 0.011   | 0.855 ± 0.005 | 0.900 ± 0.019  | 23.5         |
| ViT-S/16 (preentrenado)           |          3 | 0.853 ± 0.013 | 0.849 ± 0.008   | 0.869 ± 0.016 | 0.922 ± 0.006  | 21.7         |
| ResNet-50 (desde cero)            |          3 | 0.642 ± 0.007 | 0.687 ± 0.013   | 0.659 ± 0.003 | 0.688 ± 0.007  | 23.5         |
| ViT-S/16 (desde cero)             |          3 | 0.499 ± 0.028 | 0.550 ± 0.019   | 0.497 ± 0.029 | 0.534 ± 0.019  | 21.7         |
| EfficientNetV2B3 [Shabrina 2024]  |          1 | (0.730)†      | —               | 0.736         | —              | —            |
| MobileNetV3-Large [Shabrina 2024] |          1 | (0.713)†      | —               | 0.720         | —              | —            |
| ResNet-50 [Shabrina 2024]         |          1 | (0.675)†      | —               | 0.682         | —              | —            |

† F1 ponderado reportado por Shabrina et al. (2024), partición 90/10 sin deduplicar y 1 corrida; no es directamente comparable.
