import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polygon, Popup, CircleMarker, useMapEvents, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { loadParcelas, saveParcelas, fmtHa, fmtM2 } from '../datos';

const LAYERS = [
  { key: 'satellite', label: '🛰️ Satélite', url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}' },
  { key: 'street', label: '🗺️ Mapa', url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png' },
];

const SIERRA_CENTER = [-5.0, -79.6];   // sierra de Piura: Ayabaca, Huancabamba, Canchaque
const SIERRA_ZOOM = 9;
const CLOSE_THRESHOLD = 0.0003;
const CLIMA = { nombre: 'Huancabamba', lat: -5.2392, lon: -79.4508 };

// Códigos WMO de Open-Meteo
function condicionClima(code) {
  if (code === 0) return 'Despejado';
  if (code <= 3) return 'Parcialmente nublado';
  if (code <= 48) return 'Niebla';
  if (code <= 57) return 'Llovizna';
  if (code <= 67) return code === 65 || code === 67 ? 'Lluvia intensa' : 'Lluvia';
  if (code <= 77) return 'Nieve';
  if (code <= 82) return code === 82 ? 'Chubascos fuertes' : 'Chubascos';
  if (code <= 86) return 'Chubascos de nieve';
  return 'Tormenta';
}

function calcArea(coords) {
  if (coords.length < 3) return 0;
  const toRad = d => d * Math.PI / 180;
  const R = 6371000;
  let area = 0;
  for (let i = 0; i < coords.length; i++) {
    const j = (i + 1) % coords.length;
    area += toRad(coords[j][1] - coords[i][1]) * (2 + Math.sin(toRad(coords[i][0])) + Math.sin(toRad(coords[j][0])));
  }
  return Math.abs(area) * R * R / 2;
}

const distDeg = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const calcCenter = coords => [coords.reduce((s, c) => s + c[0], 0) / coords.length, coords.reduce((s, c) => s + c[1], 0) / coords.length];
const getFechaHoy = () => new Date().toLocaleDateString('es-PE', { day: 'numeric', month: 'short' });

function MapController({ flyTo, zoom, visible, parcelas }) {
  const map = useMap();
  const encuadrado = React.useRef(false);
  useEffect(() => { if (flyTo) map.flyTo(flyTo, zoom || 15, { duration: 1.2 }); }, [flyTo, zoom, map]);
  // el mapa se monta oculto: al mostrar la pestaña hay que recalcular su tamaño;
  // la primera vez, además, se encuadran todas las parcelas
  useEffect(() => {
    if (!visible) return;
    setTimeout(() => {
      map.invalidateSize();
      if (!encuadrado.current && parcelas.length) {
        map.fitBounds(parcelas.flatMap(p => p.coords), { padding: [25, 25] });
        encuadrado.current = true;
      }
    }, 50);
  }, [visible, map, parcelas]);
  return null;
}

function MapClickHandler({ onMapClick, drawMode }) {
  const touchCountRef = React.useRef(0);
  useMapEvents({
    touchstart: (e) => { touchCountRef.current = e.originalEvent.touches.length; },
    touchend: () => { setTimeout(() => { touchCountRef.current = 0; }, 300); },
    click: (e) => {
      if (!drawMode || touchCountRef.current > 1) return;   // ignora el zoom con dos dedos
      onMapClick(e.latlng);
    },
  });
  return null;
}

export default function Parcelas({ visible }) {
  const [parcelas, setParcelas] = useState(loadParcelas);
  const [activeLayer, setActiveLayer] = useState('satellite');
  const [showModal, setShowModal] = useState(false);
  const [editModal, setEditModal] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [drawFor, setDrawFor] = useState(null);
  const [drawPoints, setDrawPoints] = useState([]);
  const [form, setForm] = useState({ nombre: '', zona: '', hectareas: '' });
  const [flyTo, setFlyTo] = useState(null);
  const [flyZoom, setFlyZoom] = useState(15);
  const [userPos, setUserPos] = useState(null);
  const [clima, setClima] = useState({ temp: '--', condicion: 'Cargando...' });
  const [ubicandose, setUbicandose] = useState(false);

  // guardar cambios propios; recargar cuando Diagnóstico marca una parcela
  const actualizar = nuevas => { setParcelas(nuevas); saveParcelas(nuevas); };
  useEffect(() => {
    const recargar = () => setParcelas(loadParcelas());
    window.addEventListener('parcelasActualizadas', recargar);
    return () => window.removeEventListener('parcelasActualizadas', recargar);
  }, []);

  useEffect(() => {
    fetch(`https://api.open-meteo.com/v1/forecast?latitude=${CLIMA.lat}&longitude=${CLIMA.lon}&current=temperature_2m,weathercode&timezone=America%2FLima`)
      .then(r => r.json())
      .then(data => setClima({ temp: Math.round(data.current.temperature_2m), condicion: condicionClima(data.current.weathercode) }))
      .catch(() => setClima({ temp: '--', condicion: 'Sin datos' }));
  }, []);

  const volar = (centro, zoom) => { setFlyTo(centro); setFlyZoom(zoom); setTimeout(() => setFlyTo(null), 2000); };

  const irAMiUbicacion = () => {
    if (!navigator.geolocation) return;
    setUbicandose(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const p = [pos.coords.latitude, pos.coords.longitude];
        setUserPos(p);
        volar(p, 15);
        setUbicandose(false);
      },
      () => setUbicandose(false),
      { timeout: 10000 }
    );
  };

  const currentLayer = LAYERS.find(l => l.key === activeLayer);
  const alertas = parcelas.filter(p => p.estado === 'alerta').length;

  const finalizarDibujo = (points) => {
    setDrawMode(false);
    const hectareas = (calcArea(points) / 10000).toFixed(4);
    if (drawFor === 'new') {
      setForm(f => ({ ...f, hectareas }));
      setShowModal(true);
    } else if (drawFor) {
      actualizar(parcelas.map(p => p.id === drawFor ? { ...p, coords: points, hectareas: parseFloat(hectareas) } : p));
      setDrawPoints([]);
      setDrawFor(null);
    }
  };

  const handleMapClick = (latlng) => {
    const pt = [latlng.lat, latlng.lng];
    if (drawPoints.length >= 3 && distDeg(pt, drawPoints[0]) < CLOSE_THRESHOLD) {
      finalizarDibujo(drawPoints);
      return;
    }
    setDrawPoints(prev => [...prev, pt]);
  };

  const irAParcela = (parcela) => {
    volar(calcCenter(parcela.coords), 16);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const guardarParcela = () => {
    const coords = drawPoints.length >= 3 ? drawPoints
      : [[-5.2400, -79.4600], [-5.2400, -79.4570], [-5.2425, -79.4570], [-5.2425, -79.4600]];
    actualizar([...parcelas, {
      id: Date.now(), nombre: form.nombre || 'Nueva parcela', zona: form.zona || 'Piura',
      cultivo: 'Papa', emoji: '🥔', hectareas: parseFloat(form.hectareas) || 1.0,
      estado: 'sana', alerta: null, coords,
    }]);
    setForm({ nombre: '', zona: '', hectareas: '' });
    setDrawPoints([]);
    setDrawFor(null);
    setShowModal(false);
  };

  const guardarEdicion = () => {
    actualizar(parcelas.map(p => p.id === editModal.id ? {
      ...p, nombre: editModal.nombre, zona: editModal.zona, hectareas: parseFloat(editModal.hectareas) || p.hectareas,
    } : p));
    setEditModal(null);
  };

  const eliminarParcela = (id) => { actualizar(parcelas.filter(p => p.id !== id)); setEditModal(null); };

  const empezarDibujo = (para) => {
    setShowModal(false);
    setEditModal(null);
    setDrawFor(para);
    setDrawMode(true);
    setDrawPoints([]);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const areaPreview = drawPoints.length >= 3 ? calcArea(drawPoints) : null;
  const nearClose = drawPoints.length >= 3;

  return (
    <>
      <div className="header">
        <div className="logo">
          <img src="/AGROTEC.png" alt="AGROTEC" style={{ width: 32, height: 32, borderRadius: 8, objectFit: 'contain' }} />
          Parcelas
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Piura, Perú</div>
      </div>

      <div className="page">
        <div className="weather-bar">
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Hola 👋</div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>{CLIMA.nombre} · {getFechaHoy()} · {clima.temp}°C</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ fontSize: 11, color: 'rgba(29,158,117,0.9)', background: 'rgba(29,158,117,0.1)', padding: '4px 10px', borderRadius: 20 }}>
              {clima.condicion}
            </div>
            <button onClick={irAMiUbicacion}
              style={{ background: 'rgba(55,138,221,0.15)', border: '1px solid rgba(55,138,221,0.3)', color: 'var(--blue)', borderRadius: 20, padding: '4px 10px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600 }}>
              {ubicandose ? '⏳' : '📍 Ir'}
            </button>
          </div>
        </div>

        <div className="section-label">Mi cultivo</div>
        <div className="crops-scroll">
          <div className="crop-circle">
            <div className={`crop-emoji ${alertas ? 'alert' : 'active'}`}>🥔</div>
            <span className="crop-label" style={{ color: alertas ? '#ff6b6a' : 'var(--green)', fontWeight: 600 }}>Papa</span>
          </div>
          <div className="crop-circle" onClick={() => setShowModal(true)}>
            <div className="crop-emoji add">+</div>
            <span className="crop-label">Agregar</span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', alignSelf: 'center', lineHeight: 1.5, paddingLeft: 4 }}>
            {parcelas.length} parcelas · {fmtHa(parcelas.reduce((s, p) => s + Number(p.hectareas || 0), 0))} ha<br />
            {alertas ? <span style={{ color: '#ff6b6a' }}>⚠️ {alertas} con alerta</span> : <span style={{ color: '#2ecc9a' }}>✅ Todas sanas</span>}
          </div>
        </div>

        <div className="layer-btns">
          {LAYERS.map(l => (
            <button key={l.key} className={`layer-btn ${activeLayer === l.key ? 'active' : ''}`} onClick={() => setActiveLayer(l.key)}>
              {l.label}
            </button>
          ))}
        </div>

        <div className="map-wrapper">
          <MapContainer center={SIERRA_CENTER} zoom={SIERRA_ZOOM} style={{ height: '100%', width: '100%' }} zoomControl={false}>
            <TileLayer url={currentLayer.url} key={activeLayer}
              attribution={activeLayer === 'street' ? '© OpenStreetMap' : '© Esri'} />
            <MapClickHandler onMapClick={handleMapClick} drawMode={drawMode} />
            <MapController flyTo={flyTo} zoom={flyZoom} visible={visible} parcelas={parcelas} />

            {parcelas.map(p => (
              <Polygon key={p.id} positions={p.coords}
                pathOptions={{ color: p.estado === 'alerta' ? '#E24B4A' : '#1D9E75', fillColor: p.estado === 'alerta' ? '#E24B4A' : '#1D9E75', fillOpacity: 0.4, weight: 2 }}>
                <Popup>
                  <div style={{ fontFamily: 'sans-serif', minWidth: 170 }}>
                    <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{p.emoji} {p.nombre}</div>
                    <div style={{ fontSize: 12, color: '#666', marginBottom: 2 }}>📍 {p.zona}</div>
                    <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>📐 {fmtHa(p.hectareas)} ha · {fmtM2(p.hectareas)} m²</div>
                    {p.alerta
                      ? <span style={{ background: '#FCEBEB', color: '#A32D2D', padding: '2px 8px', borderRadius: 10, fontSize: 11 }}>⚠️ {p.alerta}</span>
                      : <span style={{ background: '#EAF3DE', color: '#3B6D11', padding: '2px 8px', borderRadius: 10, fontSize: 11 }}>✅ Sana</span>}
                  </div>
                </Popup>
              </Polygon>
            ))}

            {/* punto por parcela: visible aunque el mapa esté alejado; al tocarlo vuela a la parcela */}
            {!drawMode && parcelas.map(p => (
              <CircleMarker key={`c${p.id}`} center={calcCenter(p.coords)} radius={7}
                eventHandlers={{ click: () => volar(calcCenter(p.coords), 16) }}
                pathOptions={{ color: '#fff', weight: 2, fillOpacity: 1, fillColor: p.estado === 'alerta' ? '#E24B4A' : '#1D9E75' }} />
            ))}

            {drawPoints.map((pt, i) => (
              <CircleMarker key={i} center={pt} radius={i === 0 ? 9 : 7}
                pathOptions={{ color: '#fff', fillColor: i === 0 ? '#2ECC71' : i === drawPoints.length - 1 ? '#f0a030' : '#378ADD', fillOpacity: 1, weight: 2 }} />
            ))}
            {drawPoints.length >= 2 && (
              <Polygon positions={drawPoints} pathOptions={{ color: '#f0a030', fillColor: '#f0a030', fillOpacity: 0.15, weight: 2, dashArray: '6 4' }} />
            )}
            {userPos && (
              <CircleMarker center={userPos} radius={6} pathOptions={{ color: '#fff', fillColor: '#4285F4', fillOpacity: 1, weight: 2.5 }}>
                <Popup><div style={{ fontFamily: 'sans-serif' }}>📍 Tu ubicación</div></Popup>
              </CircleMarker>
            )}
          </MapContainer>
        </div>

        {drawMode && (
          <div style={{ background: 'rgba(186,117,23,0.15)', border: '1px solid rgba(186,117,23,0.4)', borderRadius: 12, padding: '10px 14px', marginBottom: 10, fontSize: 12, color: '#f0a030', textAlign: 'center' }}>
            {nearClose
              ? <span>🟢 Punto inicial = verde · <strong>Tócalo para cerrar</strong> ({drawPoints.length} puntos)</span>
              : <span>📍 Toca el mapa para marcar puntos · mínimo 3 ({drawPoints.length} puntos)</span>}
            {areaPreview && (
              <div style={{ marginTop: 6, fontSize: 14, fontWeight: 700, color: '#fff' }}>
                📐 Área: <span style={{ color: '#f0a030' }}>{fmtHa(areaPreview / 10000)} ha</span> · {fmtM2(areaPreview / 10000)} m²
              </div>
            )}
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {nearClose && (
                <button onClick={() => finalizarDibujo(drawPoints)}
                  style={{ background: 'rgba(29,158,117,0.3)', border: '1px solid var(--green)', color: 'white', borderRadius: 8, padding: '4px 14px', cursor: 'pointer', fontSize: 11, fontFamily: 'inherit', fontWeight: 600 }}>
                  ✅ Cerrar polígono
                </button>
              )}
              <button onClick={() => { setDrawMode(false); setDrawPoints([]); setDrawFor(null); }}
                style={{ background: 'transparent', border: 'none', color: '#f0a030', cursor: 'pointer', fontSize: 11, textDecoration: 'underline' }}>
                Cancelar
              </button>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
          <button className="btn-outline" style={{ flex: 1 }} onClick={() => empezarDibujo('new')}>✏️ Dibujar en mapa</button>
          <button className="btn-outline" style={{ flex: 1 }} onClick={() => setShowModal(true)}>📍 Agregar parcela</button>
        </div>

        <div className="section-label">Mis {parcelas.length} parcelas</div>
        {parcelas.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 13, padding: '20px 0' }}>Aún no tienes parcelas</div>
        )}
        {parcelas.map(p => (
          <div className="card" key={p.id} style={{ borderLeft: `3px solid ${p.estado === 'alerta' ? 'var(--red)' : 'var(--green)'}`, borderRadius: '0 14px 14px 0' }}>
            <div className="card-row">
              <div className="card-emoji-box" style={{ cursor: 'pointer' }} onClick={() => irAParcela(p)}>{p.emoji}</div>
              <div className="card-info" style={{ cursor: 'pointer' }} onClick={() => irAParcela(p)}>
                <div className="card-title">{p.nombre}</div>
                <div className="card-sub">{p.zona} · {p.cultivo}</div>
                <div className="card-sub">{fmtHa(p.hectareas)} ha · {fmtM2(p.hectareas)} m²</div>
                {p.alerta && <div className="card-sub" style={{ color: '#ff6b6a' }}>Último diagnóstico: {p.alerta}</div>}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                <span className={`badge ${p.estado === 'alerta' ? 'badge-red' : 'badge-green'}`}>
                  {p.estado === 'alerta' ? '⚠️ Alerta' : '✅ Sana'}
                </span>
                <button onClick={() => setEditModal({ ...p })}
                  style={{ background: 'transparent', border: '1px solid var(--border-light)', color: 'var(--text-secondary)', borderRadius: 8, padding: '3px 8px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit' }}>
                  ✏️ Editar
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {showModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 2000, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
          <div style={{ background: 'var(--bg-secondary)', borderRadius: '20px 20px 0 0', padding: 24, width: '100%', maxWidth: 430, maxHeight: '85vh', overflowY: 'auto' }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>📍 Nueva parcela de papa</div>
            <button onClick={() => empezarDibujo('new')}
              style={{ width: '100%', padding: 12, background: 'rgba(55,138,221,0.1)', border: '1px solid rgba(55,138,221,0.4)', color: 'var(--blue)', fontSize: 13, cursor: 'pointer', marginBottom: 16, fontFamily: 'inherit', borderRadius: 12, fontWeight: 600 }}>
              ✏️ Dibujar parcela en el mapa
            </button>
            <label className="input-label">Nombre</label>
            <input className="input-field" placeholder="Ej: Parcela Alta" value={form.nombre} onChange={e => setForm({ ...form, nombre: e.target.value })} />
            <label className="input-label">Zona / Distrito</label>
            <input className="input-field" placeholder="Ej: Huancabamba, Piura" value={form.zona} onChange={e => setForm({ ...form, zona: e.target.value })} />
            {form.hectareas && drawPoints.length >= 3 ? (
              <div style={{ background: 'rgba(29,158,117,0.1)', border: '1px solid rgba(29,158,117,0.3)', borderRadius: 10, padding: '10px 14px', marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: 'var(--green)', marginBottom: 4 }}>✅ Área calculada automáticamente del mapa</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'white' }}>{fmtHa(form.hectareas)} ha · {fmtM2(form.hectareas)} m²</div>
              </div>
            ) : null}
            <div className="grid-2">
              <div>
                <label className="input-label">Hectáreas</label>
                <input className="input-field" type="number" placeholder="Ej: 1.5" value={form.hectareas} onChange={e => setForm({ ...form, hectareas: e.target.value })} />
              </div>
              <div>
                <label className="input-label">Metros²</label>
                <input className="input-field" type="number" placeholder="Ej: 15000"
                  value={form.hectareas ? Math.round(parseFloat(form.hectareas) * 10000) : ''}
                  onChange={e => setForm({ ...form, hectareas: e.target.value ? (parseFloat(e.target.value) / 10000).toFixed(4) : '' })} />
              </div>
            </div>
            <button className="btn-primary" onClick={guardarParcela}>Guardar parcela</button>
            <button onClick={() => { setShowModal(false); setDrawPoints([]); setDrawFor(null); }}
              style={{ width: '100%', padding: 12, background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: 13, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit' }}>Cancelar</button>
          </div>
        </div>
      )}

      {editModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 2000, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
          <div style={{ background: 'var(--bg-secondary)', borderRadius: '20px 20px 0 0', padding: 24, width: '100%', maxWidth: 430, maxHeight: '85vh', overflowY: 'auto' }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>✏️ Editar parcela</div>
            <label className="input-label">Nombre</label>
            <input className="input-field" value={editModal.nombre} onChange={e => setEditModal({ ...editModal, nombre: e.target.value })} />
            <label className="input-label">Zona</label>
            <input className="input-field" value={editModal.zona} onChange={e => setEditModal({ ...editModal, zona: e.target.value })} />
            <div className="grid-2">
              <div>
                <label className="input-label">Hectáreas</label>
                <input className="input-field" type="number" value={editModal.hectareas} onChange={e => setEditModal({ ...editModal, hectareas: e.target.value })} />
              </div>
              <div>
                <label className="input-label">Metros²</label>
                <input className="input-field" type="number" value={Math.round(parseFloat(editModal.hectareas) * 10000) || ''}
                  onChange={e => setEditModal({ ...editModal, hectareas: (parseFloat(e.target.value) / 10000).toFixed(4) })} />
              </div>
            </div>
            <button className="btn-primary" onClick={guardarEdicion}>Guardar cambios</button>
            <button onClick={() => empezarDibujo(editModal.id)}
              style={{ width: '100%', padding: 12, background: 'rgba(55,138,221,0.1)', border: '1px solid rgba(55,138,221,0.3)', color: 'var(--blue)', fontSize: 13, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit', borderRadius: 12, fontWeight: 600 }}>
              🗺️ Redibujar en mapa (recalcula área)
            </button>
            <button onClick={() => eliminarParcela(editModal.id)}
              style={{ width: '100%', padding: 12, background: 'rgba(226,75,74,0.1)', border: '1px solid rgba(226,75,74,0.3)', color: '#ff6b6a', fontSize: 13, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit', borderRadius: 12, fontWeight: 600 }}>
              🗑️ Eliminar parcela
            </button>
            <button onClick={() => setEditModal(null)}
              style={{ width: '100%', padding: 12, background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: 13, cursor: 'pointer', marginTop: 4, fontFamily: 'inherit' }}>Cancelar</button>
          </div>
        </div>
      )}
    </>
  );
}
