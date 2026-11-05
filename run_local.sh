#!/usr/bin/env bash
# Cola nocturna en la PC local (GTX 1650 Ti). Orden por prioridad para el paper.
# Salta corridas terminadas (metrics.json). Si una corrida falla (p.ej. pérdida nan), sigue con la siguiente.
cd "$(dirname "$0")"
mkdir -p results/logs
export PYTHONIOENCODING=utf-8
# esperar a que termine la corrida que ya está en marcha (ESPERAR_PID). COLA se puede pasar por entorno.
while tasklist //FI "PID eq ${ESPERAR_PID:-0}" | grep -q python; do sleep 30; done
COLA="${COLA:-cnn_small:false:2 resnet50:true:0 resnet50:true:1 resnet50:true:2 resnet50:false:0 resnet50:false:1 resnet50:false:2 vit_small:false:0 vit_small:false:1 vit_small:false:2}"
for item in $COLA; do
  IFS=: read cfg pre seed <<< "$item"
  tag=$([ "$pre" = true ] && echo pre || echo scratch)
  run="papa_${cfg}_${tag}_s${seed}"
  if [ -f "results/$run/metrics.json" ]; then echo "$run ya existe"; continue; fi
  echo "===== $run $(date +%H:%M) ====="
  python src/train.py --config "configs/$cfg.yaml" --seed $seed --pretrained $pre > "results/logs/$run.log" 2>&1 \
    || echo "  FALLÓ $run (ver results/logs/$run.log)"
  grep -E "TEST|ERROR" "results/logs/$run.log" | tail -1
done
echo "COLA TERMINADA $(date +%H:%M)"
