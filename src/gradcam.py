"""Grad-CAM (Selvaraju et al., 2017) para ResNet-50, ViT-S/16 y la CNN pequeña (R5).

Uso (desde la raíz del proyecto):
    python src/gradcam.py --runs results/papa_resnet50_pre_s0 results/papa_vit_small_pre_s0 \
        --n_por_clase 2 --n_errores 8 --out results/gradcam

Para cada corrida carga best.pt y config.json, y usa test_predictions.csv para elegir imágenes de TEST:
- `aciertos_por_clase.png`: n imágenes por clase que TODOS los modelos aciertan (mismas imágenes en
  cada columna -> comparación directa CNN vs ViT).
- `errores.png`: imágenes que al menos un modelo falla, con real/predicho de cada modelo.
- `cams.npz`: mapas crudos (para métricas posteriores).

Capa objetivo (models.gradcam_target_layer):
- ResNet-50: layer4[-1] (mapa 7x7)       - ViT-S/16: blocks[-1].norm1 (tokens -> mapa 14x14)
- CNN pequeña: última conv (14x14)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES, build_model, gradcam_target_layer
from train import MEAN, STD

NOMBRE = {"resnet50": "ResNet-50", "vit_small": "ViT-S/16", "cnn_small": "CNN pequeña"}


class GradCAM:
    def __init__(self, model, capa, es_vit):
        self.model, self.es_vit = model, es_vit
        self.act = self.grad = None
        capa.register_forward_hook(self._fwd)

    def _fwd(self, _m, _inp, out):
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, clase=None):
        """x: (B,3,H,W). Devuelve (cams (B,H,W) en [0,1], logits)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        idx = logits.argmax(1) if clase is None else clase
        logits.gather(1, idx.view(-1, 1)).sum().backward()
        A, G = self.act.detach(), self.grad.detach()
        if self.es_vit:  # (B, 1+N, C) -> quitar token CLS -> (B, C, h, w)
            A, G = A[:, 1:], G[:, 1:]
            h = int(A.shape[1] ** 0.5)
            A = A.reshape(A.shape[0], h, h, -1).permute(0, 3, 1, 2)
            G = G.reshape(G.shape[0], h, h, -1).permute(0, 3, 1, 2)
        pesos = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((pesos * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn = cam.flatten(1).min(1)[0].view(-1, 1, 1)
        mx = cam.flatten(1).max(1)[0].view(-1, 1, 1)
        return ((cam - mn) / (mx - mn + 1e-8)).cpu().numpy(), logits.detach()


def cargar(run_dir, device):
    cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    n = len(CLASES[cfg["cultivo"]])
    model = build_model(cfg["model"], n, pretrained=False)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    return cfg, model


def tensor_img(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x = (torch.from_numpy(arr).permute(2, 0, 1) - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return x, arr


def superponer(img, cam, alpha=0.45):
    import matplotlib.cm as cm
    color = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * color, 0, 1)


def figura(paths, imgs, cams, info, modelos, titulo, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    filas, cols = len(paths), 1 + len(modelos)
    fig, axes = plt.subplots(filas, cols, figsize=(2.3 * cols, 2.3 * filas), squeeze=False)
    for i in range(filas):
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].set_title(f"real: {info[i]['y_true']}", fontsize=8)
        for j, m in enumerate(modelos):
            ok = info[i][m] == info[i]["y_true"]
            axes[i, j + 1].imshow(superponer(imgs[i], cams[m][i]))
            axes[i, j + 1].set_title(f"{NOMBRE.get(m, m)}: {info[i][m]}", fontsize=8,
                                     color="green" if ok else "red")
        for a in axes[i]:
            a.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / filas))
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n_por_clase", type=int, default=2)
    ap.add_argument("--n_errores", type=int, default=8)
    ap.add_argument("--out", default="results/gradcam")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    preds, modelos, cfgs = {}, [], {}
    for r in args.runs:
        cfg = json.loads((Path(r) / "config.json").read_text(encoding="utf-8"))
        m = cfg["model"]
        modelos.append(m)
        cfgs[m] = (r, cfg)
        preds[m] = pd.read_csv(Path(r) / "test_predictions.csv").set_index("path")
    base = preds[modelos[0]]
    tabla = pd.DataFrame({"y_true": base["y_true"], **{m: preds[m].loc[base.index, "y_pred"] for m in modelos}})
    todos_ok = np.logical_and.reduce([tabla[m] == tabla.y_true for m in modelos])
    alguno_mal = ~todos_ok

    rng = np.random.default_rng(args.seed)
    sel_ok = []
    for c in CLASES[cfgs[modelos[0]][1]["cultivo"]]:
        cand = tabla.index[todos_ok & (tabla.y_true == c)].tolist()
        sel_ok += list(rng.choice(cand, size=min(args.n_por_clase, len(cand)), replace=False))
    cand_err = tabla.index[alguno_mal].tolist()
    sel_err = list(rng.choice(cand_err, size=min(args.n_errores, len(cand_err)), replace=False))

    todas = sel_ok + sel_err
    cams = {}
    for m in modelos:
        r, cfg = cfgs[m]
        _, model = cargar(r, args.device)
        gc = GradCAM(model, gradcam_target_layer(model, m), es_vit=(m == "vit_small"))
        lista, imgs = [], []
        for p in todas:  # de a una imagen: entra en 4 GB aunque haya un entrenamiento corriendo
            x, arr = tensor_img(p, cfg["img_size"])
            cam, logits = gc(x[None].to(args.device))
            lista.append(cam[0])
            imgs.append(arr)
        cams[m] = np.stack(lista)
        del model, gc
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"{m}: {len(todas)} mapas Grad-CAM")

    k = len(sel_ok)
    info = tabla.loc[todas].reset_index().to_dict("records")
    figura(sel_ok, imgs[:k], {m: cams[m][:k] for m in modelos}, info[:k], modelos,
           "Grad-CAM en test — imágenes que todos los modelos aciertan", out / "aciertos_por_clase.png")
    if sel_err:
        figura(sel_err, imgs[k:], {m: cams[m][k:] for m in modelos}, info[k:], modelos,
               "Grad-CAM en test — imágenes con al menos un error (verde = acierto, rojo = error)",
               out / "errores.png")
    np.savez_compressed(out / "cams.npz", paths=np.array(todas), **{m: cams[m] for m in modelos})
    print(f"Figuras en {out}/  (aciertos: {k}, errores: {len(sel_err)}; "
          f"en test hay {int(alguno_mal.sum())} imágenes con al menos un error)")


if __name__ == "__main__":
    main()
