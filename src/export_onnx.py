"""Exporta el modelo final (ViT-S/16) a ONNX para la app web, con Grad-CAM incluido en el grafo.

El navegador (onnxruntime-web) no calcula gradientes, así que el gradiente que necesita Grad-CAM
(d logit / d salida de blocks[-1].norm1) se escribe aquí a mano. Ese gradiente solo recorre el último
bloque y la cabeza: head -> LayerNorm final -> MLP del token CLS -> atención del CLS -> claves y valores.
El resultado es idéntico al Grad-CAM de src/gradcam.py; este script lo comprueba contra PyTorch.

Uso (desde la raíz):
    python src/export_onnx.py                 # escribe web/public/modelo/vit_papa.onnx (+ versión int8)
    python src/export_onnx.py --evaluar       # además mide macro-F1 en test con onnxruntime (fp32 e int8)

Salidas del grafo, para 1 imagen (1,3,224,224) normalizada con MEAN/STD de ImageNet:
    prob (1,7)        probabilidades softmax, orden de CLASES["papa"]
    cam  (1,224,224)  Grad-CAM de la clase predicha, en [0,1]
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES, build_model, gradcam_target_layer
from gradcam import GradCAM
from train import MEAN, STD

RAIZ = Path(__file__).resolve().parent.parent


def ln_back(x, ln, gamma, g):
    """Gradiente de LayerNorm respecto a su entrada x, dado el gradiente g de su salida."""
    mu = x.mean(-1, keepdim=True)
    sigma = torch.sqrt(((x - mu) ** 2).mean(-1, keepdim=True) + ln.eps)
    xhat = (x - mu) / sigma
    gx = g * gamma
    return (gx - gx.mean(-1, keepdim=True) - xhat * (gx * xhat).mean(-1, keepdim=True)) / sigma


def gelu_prima(h):
    return 0.5 * (1 + torch.erf(h / math.sqrt(2))) + h * torch.exp(-0.5 * h * h) / math.sqrt(2 * math.pi)


class ViTConCAM(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.lado = m.patch_embed.grid_size[0]          # 14 parches por lado
        b = m.blocks[-1]
        assert m.global_pool == "token" and isinstance(b.ls1, nn.Identity) and isinstance(b.attn.norm, nn.Identity)
        assert getattr(b.attn, "gate", None) is None and isinstance(b.attn.q_norm, nn.Identity)
        # copias propias de los pesos que usa el gradiente: si se comparten con las capas del forward,
        # la cuantización int8 reemplaza el inicializador compartido y el grafo queda inválido
        C = m.embed_dim
        for nombre, w in {"w_head": m.head.weight, "w_fc2": b.mlp.fc2.weight, "w_fc1": b.mlp.fc1.weight,
                          "w_proj": b.attn.proj.weight, "w_k": b.attn.qkv.weight[C:2 * C],
                          "w_v": b.attn.qkv.weight[2 * C:], "g_norm": m.norm.weight, "g_norm2": b.norm2.weight}.items():
            self.register_buffer(nombre, w.detach().clone())

    def forward(self, x):
        m = self.m
        t = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
        for blk in m.blocks[:-1]:
            t = blk(t)
        b, at = m.blocks[-1], m.blocks[-1].attn
        A = b.norm1(t)                                         # capa objetivo de Grad-CAM (1, 197, 384)
        _, N, C = A.shape
        H = at.num_heads
        d = C // H
        qkv = at.qkv(A).reshape(1, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (1, H, N, d)
        P = ((q * at.scale) @ k.transpose(-2, -1)).softmax(-1)
        y = t + at.proj((P @ v).transpose(1, 2).reshape(1, N, C))
        h1 = b.mlp.fc1(b.norm2(y))
        z = y + b.mlp.fc2(b.mlp.act(h1))
        logits = m.head(m.norm(z)[:, 0])
        prob = logits.softmax(-1)

        # gradiente de logit[clase predicha] respecto a A (solo importa el camino del token CLS)
        c = logits.argmax(-1)
        g_z = ln_back(z[:, 0], m.norm, self.g_norm, self.w_head.index_select(0, c))
        g_n2 = ((g_z @ self.w_fc2) * gelu_prima(h1[:, 0])) @ self.w_fc1
        g_y = g_z + ln_back(y[:, 0], b.norm2, self.g_norm2, g_n2)
        g_o = (g_y @ self.w_proj).reshape(1, H, 1, d)       # gradiente de la salida de atención del CLS
        Pc = P[:, :, 0, :].unsqueeze(-1)                       # (1, H, N, 1) atención del CLS a cada token
        gov = (v * g_o).sum(-1, keepdim=True)
        g_s = Pc * (gov - (Pc * gov).sum(2, keepdim=True))     # derivada del softmax
        g_k = g_s * (q[:, :, :1, :] * at.scale)
        g_v = Pc * g_o
        Wk, Wv = self.w_k, self.w_v
        G = g_k.permute(0, 2, 1, 3).reshape(1, N, C) @ Wk + g_v.permute(0, 2, 1, 3).reshape(1, N, C) @ Wv

        Ap, Gp = A[:, 1:], G[:, 1:]                            # sin el token CLS -> 14x14 parches
        lado = self.lado
        cam = F.relu((Gp.mean(1, keepdim=True) * Ap).sum(-1)).reshape(1, 1, lado, lado)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn = cam.amin(dim=(1, 2), keepdim=True)
        mx = cam.amax(dim=(1, 2), keepdim=True)
        return prob, (cam - mn) / (mx - mn + 1e-8)


def tensor_img(path, size=224):
    """Igual que la evaluación de train.py: Resize((224,224)) + normalización."""
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return tf(Image.open(path).convert("RGB"))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "modelo_final"))
    ap.add_argument("--out", default=str(RAIZ / "web/public/modelo"))
    ap.add_argument("--evaluar", action="store_true")
    args = ap.parse_args()

    carpeta, out = Path(args.modelo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    clases = CLASES[cfg["cultivo"]]
    m = build_model(cfg["model"], len(clases), pretrained=False)
    m.load_state_dict(torch.load(carpeta / "best.pt", map_location="cpu", weights_only=True))
    m.eval()
    envoltura = ViTConCAM(m).eval()

    # 1) el grafo manual coincide con PyTorch (probabilidades y Grad-CAM)
    m_ref = copy.deepcopy(m)   # copia aparte: GradCAM deja un hook en norm1
    ref_cam = GradCAM(m_ref, gradcam_target_layer(m_ref, cfg["model"]), es_vit=True)
    ejemplos = sorted((RAIZ / "demo/ejemplos").glob("*.jpg"))
    for p in ejemplos:
        x = tensor_img(p)
        cam_ref, logits_ref = ref_cam(x)
        with torch.no_grad():
            prob, cam = envoltura(x)
        dp = (prob - logits_ref.softmax(-1)).abs().max().item()
        dc = np.abs(cam.numpy() - cam_ref).max()
        print(f"  {p.stem:13s} pred={clases[int(prob.argmax())]:13s} |Δprob|={dp:.1e} |Δcam|={dc:.1e}")
        assert dp < 1e-4 and dc < 1e-3, "el Grad-CAM manual no coincide con el de PyTorch"

    # 2) exportar
    onnx32 = carpeta / "vit_papa_fp32.onnx"     # intermedio (89 MB): no va a la web
    torch.onnx.export(envoltura, tensor_img(ejemplos[0]), str(onnx32), input_names=["imagen"],
                      output_names=["prob", "cam"], opset_version=17, do_constant_folding=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    onnx8 = out / "vit_papa_int8.onnx"
    quantize_dynamic(str(onnx32), str(onnx8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])   # la conv de parches queda en fp32
    for f in (onnx32, onnx8):
        print(f"  {f.name}: {f.stat().st_size / 2**20:.1f} MB")
    (out / "clases.json").write_text(json.dumps(clases), encoding="utf-8")

    # 3) opcional: macro-F1 en test con onnxruntime
    if args.evaluar:
        import onnxruntime as ort
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score
        df = pd.read_csv(RAIZ / cfg["split_csv"])
        df = df[df.split == "test"]
        y = df.label.map({c: i for i, c in enumerate(clases)}).to_numpy()
        for f in (onnx32, onnx8):
            s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
            pred = np.array([s.run(["prob"], {"imagen": tensor_img(RAIZ / p).numpy()})[0].argmax()
                             for p in df.path])
            print(f"  {f.name}: test macro-F1 {f1_score(y, pred, average='macro'):.4f} · "
                  f"accuracy {accuracy_score(y, pred):.4f}  (PyTorch: 0.8645 · 0.8830)")


if __name__ == "__main__":
    main()
