"""Demo web: sube una foto de hoja de papa -> enfermedad + confianza + Grad-CAM.

Uso local (desde la raíz del proyecto):
    python src/demo_app.py                 # abre http://127.0.0.1:7860
    python src/demo_app.py --share         # además crea un enlace público temporal (gradio.live)

Carga el modelo final de modelo_final/ (elegido por validación con src/select_final.py).
Si falta modelo_final/best.pt, lo descarga del Release de GitHub (PESOS_URL).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models import CLASES, build_model, gradcam_target_layer
from gradcam import GradCAM, superponer
from train import MEAN, STD

RAIZ = Path(__file__).resolve().parent.parent
PESOS_URL = "https://github.com/gabo2607/papa-cnn-vs-vit/releases/download/modelo-final/best.pt"
NOMBRES = {
    "bacteria": "Bacteriosis (bacteria)",
    "fungi": "Hongos (fungi)",
    "healthy": "Sana",
    "nematode": "Nematodos",
    "pest": "Plaga de insectos (pest)",
    "phytophthora": "Rancha / tizón tardío (Phytophthora)",
    "virus": "Virosis (virus)",
}
MODELO_TXT = {"vit_small": "ViT-S/16 (Vision Transformer)", "resnet50": "ResNet-50 (CNN)",
              "cnn_small": "CNN pequeña"}


def asegurar_pesos(carpeta):
    """Descarga best.pt (≈83 MB) del Release de GitHub si no está en la carpeta."""
    pt = Path(carpeta) / "best.pt"
    if not pt.exists():
        print(f"Descargando pesos del modelo final desde {PESOS_URL} ...")
        torch.hub.download_url_to_file(PESOS_URL, str(pt), progress=True)
    return pt


def cargar_modelo(carpeta, device):
    carpeta = Path(carpeta)
    asegurar_pesos(carpeta)
    cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8")) if (carpeta / "meta.json").exists() else {}
    clases = CLASES[cfg.get("cultivo", "papa")]
    model = build_model(cfg["model"], len(clases), pretrained=False)
    model.load_state_dict(torch.load(carpeta / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    cam = GradCAM(model, gradcam_target_layer(model, cfg["model"]), es_vit=(cfg["model"] == "vit_small"))
    return cfg, meta, clases, model, cam


def preparar(img: Image.Image, size: int):
    """Recorte cuadrado central (el dataset es 1:1) y redimensionado a size x size."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = (torch.from_numpy(arr).permute(2, 0, 1) - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return x[None], arr


def construir_app(carpeta, device):
    import gradio as gr
    cfg, meta, clases, model, cam = cargar_modelo(carpeta, device)

    def predecir(img):
        if img is None:
            return None, None, ""
        x, arr = preparar(img, cfg["img_size"])
        mapa, logits = cam(x.to(device))
        prob = logits.float().softmax(1)[0].cpu().numpy()
        top = int(prob.argmax())
        etiquetas = {NOMBRES[c]: float(p) for c, p in zip(clases, prob)}
        overlay = (superponer(arr, mapa[0]) * 255).astype(np.uint8)
        aviso = ""
        if prob[top] < 0.6:
            aviso = (f"⚠️ Confianza baja ({prob[top]:.0%}). La foto puede no parecerse a las de entrenamiento "
                     "(hoja de papa en campo, de cerca). Tómela como orientativa.")
        return etiquetas, overlay, aviso

    t = meta.get("test", {})
    desc = (f"**Modelo:** {MODELO_TXT.get(cfg['model'], cfg['model'])} preentrenado en ImageNet y ajustado con "
            f"3,015 fotos de campo (Java Central, Indonesia).  \n"
            + (f"**Desempeño en prueba (imágenes nunca vistas):** macro-F1 {t['macro_f1']:.3f} · "
               f"accuracy {t['accuracy']:.1%}.  \n" if t else "")
            + "El **mapa Grad-CAM** muestra en rojo las zonas de la foto que más pesaron en la decisión.  \n"
              "*Demo académica (MIA-203, UNI). No reemplaza el diagnóstico de un agrónomo.*")
    ejemplos = sorted(str(p) for p in (RAIZ / "demo" / "ejemplos").glob("*") if p.is_file())

    with gr.Blocks(title="Enfermedades en hojas de papa") as app:
        gr.Markdown("# 🥔 Detección de enfermedades en hojas de papa\n" + desc)
        with gr.Row():
            with gr.Column():
                entrada = gr.Image(type="pil", label="Foto de la hoja", height=320)
                boton = gr.Button("Diagnosticar", variant="primary")
            with gr.Column():
                salida = gr.Label(num_top_classes=3, label="Predicción")
                aviso = gr.Markdown()
                mapa = gr.Image(label="Grad-CAM: ¿dónde miró el modelo?", height=320)
        if ejemplos:
            gr.Examples(ejemplos, inputs=entrada, label="Ejemplos del conjunto de prueba (haz clic en uno)")
        boton.click(predecir, entrada, [salida, mapa, aviso])
        entrada.change(predecir, entrada, [salida, mapa, aviso])  # al subir foto o elegir un ejemplo
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "modelo_final"))
    ap.add_argument("--device", default="cpu", help="cpu por defecto (la inferencia de 1 imagen es rápida)")
    ap.add_argument("--share", action="store_true", help="enlace público temporal de gradio")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    # server_name explícito: con el valor por defecto Gradio no detecta localhost en algunas PCs Windows
    construir_app(args.modelo, args.device).launch(share=args.share, server_name="127.0.0.1",
                                                   server_port=args.port, inbrowser=not args.share)


if __name__ == "__main__":
    main()
