// Modelo final del proyecto (ViT-S/16 preentrenado, semilla 2) corriendo en el navegador con onnxruntime-web.
// El .onnx lo genera src/export_onnx.py: devuelve las probabilidades de las 7 clases y el mapa Grad-CAM.
import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;

const URL_MODELO = '/modelo/vit_papa_int8.onnx';
const LADO = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
export const UMBRAL_CONFIANZA = 0.6;

// Orden de salida del modelo = CLASES["papa"] de src/models.py
export const CLASES = ['bacteria', 'fungi', 'healthy', 'nematode', 'pest', 'phytophthora', 'virus'];

export const INFO = {
  bacteria: {
    nombre: 'Bacteriosis', emoji: '🦠',
    descripcion: 'Daño causado por bacterias: lesiones de aspecto acuoso, manchas oscuras o marchitez de hojas y tallos.',
    recomendacion: 'Retire y elimine las plantas afectadas fuera de la parcela, desinfecte herramientas, evite el exceso de riego y use semilla certificada.',
  },
  fungi: {
    nombre: 'Enfermedad por hongos', emoji: '🍄',
    descripcion: 'Manchas marrones o negras en la hoja, a veces con anillos concéntricos (por ejemplo, tizón temprano por Alternaria).',
    recomendacion: 'Retire las hojas afectadas, mejore la ventilación entre plantas y rote cultivos. Consulte a un agrónomo antes de aplicar un fungicida.',
  },
  healthy: {
    nombre: 'Hoja sana', emoji: '✅', sana: true,
    descripcion: 'No se observan síntomas de enfermedad ni de plaga en la hoja.',
    recomendacion: 'Siga monitoreando la parcela cada semana, sobre todo después de lluvias o heladas.',
  },
  nematode: {
    nombre: 'Daño por nematodos', emoji: '🪱',
    descripcion: 'Los nematodos atacan la raíz y el tubérculo; en la hoja se ve amarillamiento y plantas débiles o enanas.',
    recomendacion: 'Revise raíces y tubérculos, haga un análisis de suelo, rote con cultivos que no sean hospederos y use semilla sana.',
  },
  pest: {
    nombre: 'Plaga de insectos', emoji: '🐛',
    descripcion: 'Hojas comidas, perforadas o con minas, por insectos como pulgones, minadores o larvas.',
    recomendacion: 'Inspeccione el envés de las hojas, use trampas para monitorear y aplique manejo integrado de plagas.',
  },
  phytophthora: {
    nombre: 'Rancha / tizón tardío', emoji: '⚠️',
    descripcion: 'Causada por Phytophthora infestans: manchas oscuras de aspecto húmedo y, con humedad, moho blanco en el envés. Se propaga muy rápido con clima frío y húmedo.',
    recomendacion: 'Actúe rápido: elimine el follaje afectado, avise a las parcelas vecinas y consulte a un agrónomo para el control químico.',
  },
  virus: {
    nombre: 'Virosis', emoji: '🧬',
    descripcion: 'Mosaicos, enrollamiento, deformación o amarillamiento de hojas. La transmiten pulgones y la semilla infectada.',
    recomendacion: 'No tiene cura química: elimine las plantas enfermas, controle los pulgones y use semilla certificada.',
  },
};

export const META = { modelo: 'ViT-S/16 (Vision Transformer)', macroF1: 0.864, accuracy: 0.883, mb: 24 };

let sesion = null;
export function cargarModelo() {
  if (!sesion) {
    sesion = ort.InferenceSession.create(URL_MODELO, { executionProviders: ['wasm'] })
      .catch(e => { sesion = null; throw e; });
  }
  return sesion;
}

export function cargarImagen(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Recorte cuadrado central (el dataset es 1:1) y redimensionado, como en src/demo_app.py
function recorte(img, lado) {
  const c = document.createElement('canvas');
  c.width = c.height = lado;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  const s = Math.min(img.naturalWidth, img.naturalHeight);
  ctx.drawImage(img, (img.naturalWidth - s) / 2, (img.naturalHeight - s) / 2, s, s, 0, 0, lado, lado);
  return c;
}

function aTensor(canvas) {
  const { data } = canvas.getContext('2d').getImageData(0, 0, LADO, LADO);
  const n = LADO * LADO;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) x[c * n + i] = (data[4 * i + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor('float32', x, [1, 3, LADO, LADO]);
}

const jet = v => [1.5 - Math.abs(4 * v - 3), 1.5 - Math.abs(4 * v - 2), 1.5 - Math.abs(4 * v - 1)]
  .map(t => Math.round(255 * Math.min(1, Math.max(0, t))));

// Foto + Grad-CAM (colormap jet, 45 % de opacidad, como en src/gradcam.py)
function superponer(img, cam) {
  const camC = document.createElement('canvas');
  camC.width = camC.height = LADO;
  const ctx = camC.getContext('2d');
  const im = ctx.createImageData(LADO, LADO);
  for (let i = 0; i < LADO * LADO; i++) {
    const [r, g, b] = jet(cam[i]);
    im.data.set([r, g, b, 255], 4 * i);
  }
  ctx.putImageData(im, 0, 0);
  const out = recorte(img, 448);
  const octx = out.getContext('2d');
  octx.globalAlpha = 0.45;
  octx.drawImage(camC, 0, 0, 448, 448);
  return out.toDataURL('image/jpeg', 0.85);
}

export async function diagnosticar(img) {
  const s = await cargarModelo();
  const t0 = performance.now();
  const salida = await s.run({ imagen: aTensor(recorte(img, LADO)) });
  const ms = Math.round(performance.now() - t0);
  const prob = Array.from(salida.prob.data);
  const ranking = CLASES.map((c, i) => ({ clase: c, p: prob[i] })).sort((a, b) => b.p - a.p);
  return { ranking, top: ranking[0], camUrl: superponer(img, salida.cam.data), ms };
}
