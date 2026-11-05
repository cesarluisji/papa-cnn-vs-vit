"""Prueba rápida: construye cada modelo, cuenta parámetros y mide un paso de entrenamiento en GPU."""
import time
import torch
import torch.nn as nn

from models import build_model, count_params, CLASES

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}\n")

BATCH = 16
num_classes = len(CLASES["papa"])
x = torch.randn(BATCH, 3, 224, 224, device=device)
y = torch.randint(0, num_classes, (BATCH,), device=device)

print(f"{'modelo':<10} {'params (M)':>10} {'salida':>14} {'ms/paso':>9} {'VRAM pico (MB)':>15}")
for name in ["cnn_small", "resnet50", "vit_small"]:
    model = build_model(name, num_classes, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(6):
        t0 = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= 2:  # descartar calentamiento
            times.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
    print(f"{name:<10} {count_params(model)/1e6:>10.2f} {str(tuple(out.shape)):>14} "
          f"{sum(times)/len(times):>9.0f} {mem:>15.0f}")
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
