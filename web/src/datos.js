// Datos compartidos entre pantallas. Todo se guarda en el localStorage del navegador (sin login).
export const STORAGE_KEY = 'papa_parcelas';

// Parcelas de ejemplo en zonas paperas de la sierra de Piura
export const INITIAL_PARCELAS = [
  { id: 1, nombre: 'Parcela Alta', zona: 'Huancabamba, Piura', cultivo: 'Papa', emoji: '🥔', hectareas: 1.2, estado: 'sana', alerta: null,
    coords: [[-5.2380, -79.4530], [-5.2380, -79.4490], [-5.2410, -79.4490], [-5.2410, -79.4530]] },
  { id: 2, nombre: 'Parcela Ayabaca', zona: 'Ayabaca, Piura', cultivo: 'Papa', emoji: '🥔', hectareas: 0.8, estado: 'sana', alerta: null,
    coords: [[-4.6380, -79.7180], [-4.6380, -79.7150], [-4.6405, -79.7150], [-4.6405, -79.7180]] },
  { id: 3, nombre: 'Parcela Canchaque', zona: 'Canchaque, Piura', cultivo: 'Papa', emoji: '🥔', hectareas: 0.5, estado: 'sana', alerta: null,
    coords: [[-5.3720, -79.6050], [-5.3720, -79.6030], [-5.3740, -79.6030], [-5.3740, -79.6050]] },
];

export function loadParcelas() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : INITIAL_PARCELAS;
  } catch { return INITIAL_PARCELAS; }
}

export function saveParcelas(p) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); } catch { /* almacenamiento lleno o bloqueado */ }
  window.dispatchEvent(new CustomEvent('parcelasActualizadas'));
}

// Al guardar un diagnóstico, la parcela pasa a "Alerta" (o "Sana") según el resultado
export function marcarParcela(id, enfermedad, sana) {
  saveParcelas(loadParcelas().map(p => p.id === id
    ? { ...p, estado: sana ? 'sana' : 'alerta', alerta: sana ? null : enfermedad } : p));
}

// Mismo formato de números (es-PE) para hectáreas y m²
export const fmtHa = ha => Number(ha).toLocaleString('es-PE', { maximumFractionDigits: 2 });
export const fmtM2 = ha => Math.round(Number(ha) * 10000).toLocaleString('es-PE');

export const comprimirImagen = (src) => new Promise((resolve) => {
  const img = new Image();
  img.onload = () => {
    const MAX = 600;
    let w = img.width, h = img.height;
    if (w > MAX) { h = Math.round((h * MAX) / w); w = MAX; }
    if (h > MAX) { w = Math.round((w * MAX) / h); h = MAX; }
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(img, 0, 0, w, h);
    resolve(canvas.toDataURL('image/jpeg', 0.5));
  };
  img.src = src;
});
