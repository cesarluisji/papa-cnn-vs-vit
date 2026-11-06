import React, { useState, useRef, useEffect } from 'react';
import { CLASES, INFO, META, UMBRAL_CONFIANZA, cargarModelo, cargarImagen, diagnosticar } from '../modelo';
import { loadParcelas, marcarParcela, comprimirImagen } from '../datos';

const EJEMPLOS = CLASES.map(c => ({ clase: c, src: `/ejemplos/${c}.jpg` }));
const pct = p => `${Math.round(p * 100)}%`;

export default function Diagnostico() {
  const [estadoModelo, setEstadoModelo] = useState('cargando'); // cargando | listo | error
  const [parcelas, setParcelas] = useState(loadParcelas);
  const [parcelaOpen, setParcelaOpen] = useState(false);
  const [selectedParcelaId, setSelectedParcelaId] = useState(null);
  const [image, setImage] = useState(null);         // URL de la foto
  const [ejemplo, setEjemplo] = useState(null);     // clase real si es foto de ejemplo
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [verMapa, setVerMapa] = useState(true);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef();

  const parcelaSeleccionada = parcelas.find(p => p.id === selectedParcelaId) || parcelas[0];

  useEffect(() => {
    cargarModelo().then(() => setEstadoModelo('listo')).catch(() => setEstadoModelo('error'));
    const recargar = () => setParcelas(loadParcelas());
    window.addEventListener('parcelasActualizadas', recargar);
    return () => window.removeEventListener('parcelasActualizadas', recargar);
  }, []);

  const reiniciar = () => { setResult(null); setSaved(false); setError(null); };

  const analizar = async (src) => {
    setLoading(true);
    setError(null);
    try {
      const img = await cargarImagen(src);
      setResult(await diagnosticar(img));
      setEstadoModelo('listo');
      setVerMapa(true);
    } catch (e) {
      setError('No se pudo analizar la foto. Revisa tu conexión (la primera vez se descarga el modelo) y reintenta.');
    }
    setLoading(false);
  };

  const handleImage = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      setEjemplo(null);
      reiniciar();
    }
    e.target.value = '';
  };

  const usarEjemplo = (ej) => {
    setImage(ej.src);
    setEjemplo(ej.clase);
    reiniciar();
    analizar(ej.src);
  };

  const guardar = async () => {
    if (!result) return;
    try {
      const ahora = new Date();
      const info = INFO[result.top.clase];
      const diagData = {
        id: Date.now(),
        timestamp: ahora.getTime(),
        fecha: ahora.toLocaleDateString('es-PE', { day: '2-digit', month: 'short', year: 'numeric' }) + ' ' +
          ahora.toLocaleTimeString('es-PE', { hour: '2-digit', minute: '2-digit' }),
        cultivo: 'Papa',
        parcelaId: parcelaSeleccionada?.id || null,
        foto: await comprimirImagen(image),
        resultado: {
          clase: result.top.clase, enfermedad: info.nombre, confianza: Math.round(result.top.p * 100),
          descripcion: info.descripcion, recomendacion: info.recomendacion, sana: !!info.sana,
        },
      };
      localStorage.setItem(`diag_${diagData.id}`, JSON.stringify(diagData));
      if (parcelaSeleccionada) marcarParcela(parcelaSeleccionada.id, info.nombre, info.sana);
      window.dispatchEvent(new CustomEvent('diagGuardado'));
      setSaved(true);
    } catch (err) {
      setError('No se pudo guardar (¿almacenamiento del navegador lleno?).');
    }
  };

  const nuevoDiagnostico = () => { setImage(null); setEjemplo(null); reiniciar(); };

  const top = result?.top;
  const info = top ? INFO[top.clase] : null;
  const baja = top && top.p < UMBRAL_CONFIANZA;

  return (
    <>
      <div className="header">
        <div className="logo">
          <img src="/AGROTEC.png" alt="AGROTEC" style={{ width: 32, height: 32, borderRadius: 8, objectFit: 'contain' }} />
          Diagnóstico IA
        </div>
        <div style={{ fontSize: 20 }}>🔍</div>
      </div>

      <div className="page">
        <div className="weather-bar" style={{ alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>🥔 Cultivo: Papa</div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
              {META.modelo} · macro-F1 {META.macroF1.toFixed(3)} · {pct(META.accuracy)} de aciertos en prueba
            </div>
          </div>
          <div style={{ fontSize: 11, padding: '4px 10px', borderRadius: 20, whiteSpace: 'nowrap',
            color: estadoModelo === 'listo' ? '#2ecc9a' : estadoModelo === 'error' ? '#ff6b6a' : '#f0a030',
            background: estadoModelo === 'listo' ? 'rgba(29,158,117,0.1)' : estadoModelo === 'error' ? 'rgba(226,75,74,0.1)' : 'rgba(186,117,23,0.1)' }}>
            {estadoModelo === 'listo' ? '● Modelo listo' : estadoModelo === 'error' ? '● Sin modelo' : `⏳ Cargando (${META.mb} MB)`}
          </div>
        </div>

        <input type="file" accept="image/*" capture="environment" ref={fileRef} onChange={handleImage} style={{ display: 'none' }} />

        {!image ? (
          <>
            <div className="upload-zone" onClick={() => fileRef.current.click()}>
              <div className="icon">📷</div>
              <div className="title">Foto de la hoja de papa</div>
              <div className="sub">Tomar foto o subir desde galería · una hoja, de cerca</div>
            </div>
            <div className="section-label">O prueba con una foto de ejemplo</div>
            <div className="crops-scroll" style={{ marginBottom: 12 }}>
              {EJEMPLOS.map(ej => (
                <div className="crop-circle" key={ej.clase} onClick={() => usarEjemplo(ej)}>
                  <img src={ej.src} alt={INFO[ej.clase].nombre}
                    style={{ width: 56, height: 56, borderRadius: 12, objectFit: 'cover', border: '2px solid var(--border-light)' }} />
                  <span className="crop-label" style={{ maxWidth: 64 }}>{INFO[ej.clase].nombre}</span>
                </div>
              ))}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginBottom: 14 }}>
              Fotos del conjunto de prueba: el modelo nunca las vio durante el entrenamiento.
            </div>
          </>
        ) : (
          <div style={{ marginBottom: 14, position: 'relative' }}>
            <img src={result && verMapa ? result.camUrl : image} alt="hoja"
              style={{ width: '100%', aspectRatio: '1 / 1', borderRadius: 14, objectFit: 'cover', border: '1px solid var(--border)', display: 'block' }} />
            <button onClick={nuevoDiagnostico}
              style={{ position: 'absolute', top: 8, right: 8, background: 'rgba(0,0,0,0.6)', border: 'none', color: 'white', borderRadius: 20, padding: '4px 10px', cursor: 'pointer', fontSize: 12 }}>
              ✕ Cambiar
            </button>
            {result && (
              <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                <button className={`layer-btn ${!verMapa ? 'active' : ''}`} onClick={() => setVerMapa(false)}>📷 Foto</button>
                <button className={`layer-btn ${verMapa ? 'active' : ''}`} onClick={() => setVerMapa(true)}>🔥 ¿Dónde miró la IA?</button>
              </div>
            )}
            {result && verMapa && (
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 6 }}>
                Mapa Grad-CAM: en rojo, las zonas de la hoja que más pesaron en la decisión.
              </div>
            )}
          </div>
        )}

        <label className="input-label">Selecciona parcela</label>
        <div className="dropdown">
          <div className="dropdown-header" onClick={() => setParcelaOpen(!parcelaOpen)}>
            <span>{parcelaSeleccionada ? `${parcelaSeleccionada.emoji} ${parcelaSeleccionada.nombre}${parcelaSeleccionada.zona ? ` — ${parcelaSeleccionada.zona}` : ''}` : 'Sin parcelas'}</span>
            <span>{parcelaOpen ? '▲' : '▼'}</span>
          </div>
          {parcelaOpen && parcelas.map(p => (
            <div key={p.id} className={`dropdown-option ${p.id === parcelaSeleccionada?.id ? 'selected' : ''}`}
              onClick={() => { setSelectedParcelaId(p.id); setParcelaOpen(false); }}>
              {p.emoji} {p.nombre}{p.zona ? ` — ${p.zona}` : ''}
            </div>
          ))}
        </div>

        {error && (
          <div className="result-warning" style={{ marginBottom: 10 }}>
            <div className="result-title" style={{ color: 'var(--red)' }}>❌ {error}</div>
            <button onClick={() => analizar(image)}
              style={{ marginTop: 8, background: 'rgba(226,75,74,0.2)', border: '1px solid var(--red)', color: 'var(--red)', borderRadius: 8, padding: '6px 14px', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', width: '100%' }}>
              🔄 Reintentar
            </button>
          </div>
        )}

        {result && (
          <>
            {baja && (
              <div className="result-warning">
                <div className="result-title" style={{ color: '#f0a030' }}>⚠️ Confianza baja ({pct(top.p)})</div>
                <div className="result-sub">
                  La foto puede no parecerse a las de entrenamiento (una hoja de papa en campo, de cerca).
                  Tome el resultado como orientativo o pruebe con otra foto.
                </div>
              </div>
            )}

            <div className={info.sana ? 'result-success' : 'result-warning'}>
              <div className="result-title" style={{ color: info.sana ? 'var(--green)' : '#f0a030', fontSize: 15 }}>
                {info.emoji} {info.nombre}
              </div>
              <div className="result-sub">Confianza: {pct(top.p)} · analizado en tu dispositivo en {result.ms} ms</div>
              <div className="result-sub" style={{ marginTop: 6 }}>{info.descripcion}</div>
              {ejemplo && (
                <div className="result-sub" style={{ marginTop: 6 }}>
                  Etiqueta real de esta foto de ejemplo: <strong style={{ color: 'white' }}>{INFO[ejemplo].nombre}</strong>{' '}
                  {ejemplo === top.clase ? '✅' : '❌'}
                </div>
              )}
            </div>

            <div className="card">
              <div className="section-label" style={{ marginBottom: 8 }}>Las 3 más probables</div>
              {result.ranking.slice(0, 3).map((r, i) => (
                <div key={r.clase} style={{ marginBottom: i < 2 ? 8 : 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
                    <span>{INFO[r.clase].emoji} {INFO[r.clase].nombre}</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{pct(r.p)}</span>
                  </div>
                  <div style={{ height: 6, background: 'var(--bg-card2)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.max(1, r.p * 100)}%`, background: i === 0 ? 'var(--green)' : 'var(--text-tertiary)', borderRadius: 3 }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="result-info">
              <div className="result-title" style={{ color: 'var(--blue)' }}>🌱 ¿Qué hacer?</div>
              <div className="result-sub">{info.recomendacion}</div>
              <div className="result-sub" style={{ marginTop: 6, fontSize: 11, color: 'var(--text-tertiary)' }}>
                Orientación general. No reemplaza la visita de un agrónomo.
              </div>
            </div>

            <button className="btn-primary" onClick={guardar} disabled={saved}
              style={{ background: saved ? 'var(--text-tertiary)' : 'var(--green)' }}>
              {saved ? '✅ Guardado en historial' : '💾 Guardar en historial'}
            </button>
            {saved && (
              <button onClick={() => window.dispatchEvent(new CustomEvent('goToHistorial'))}
                style={{ width: '100%', padding: 12, background: 'rgba(55,138,221,0.1)', border: '1px solid rgba(55,138,221,0.4)', color: 'var(--blue)', fontSize: 13, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit', borderRadius: 12, fontWeight: 600 }}>
                📅 Ver en el historial
              </button>
            )}
            <button onClick={nuevoDiagnostico}
              style={{ width: '100%', padding: 12, background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-secondary)', fontSize: 13, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit', borderRadius: 12 }}>
              🔄 Nuevo diagnóstico
            </button>
          </>
        )}

        {!result && (
          <button className="btn-primary" onClick={() => analizar(image)} disabled={!image || loading}
            style={{ opacity: !image ? 0.5 : 1 }}>
            {loading ? '🔄 Analizando...' : '🔍 Diagnosticar ahora'}
          </button>
        )}

        <div style={{ fontSize: 10, color: 'var(--text-tertiary)', textAlign: 'center', margin: '16px 0 4px', lineHeight: 1.5 }}>
          La foto no se envía a ningún servidor: el modelo corre en tu navegador.<br />
          Proyecto final MIA-203 · Redes Neuronales y Aprendizaje Profundo · UNI FIIS
        </div>
      </div>
    </>
  );
}
