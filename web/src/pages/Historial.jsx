import React, { useState, useEffect } from 'react';
import { loadParcelas } from '../datos';

// fecha local (no UTC) en formato AAAA-MM-DD
const isoLocal = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
const haceDias = (n) => isoLocal(new Date(Date.now() - n * 86400000));

const MESES = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];

const TIPO_COLORES = {
  'Riego': '#378ADD',
  'Fertilización': '#1D9E75',
  'Fumigación': '#BA7517',
  'Cosecha': '#9B59B6',
  'Siembra': '#2ECC71',
  'Manual': '#888',
  'IA': '#E24B4A',
};

function getDaysInMonth(year, month) {
  return new Date(year, month + 1, 0).getDate();
}

function getFirstDayOfMonth(year, month) {
  let day = new Date(year, month, 1).getDay();
  return day === 0 ? 6 : day - 1;
}

export default function Historial() {
  // Leer parcelas dinámicamente del localStorage
  const [parcelas, setParcelas] = useState(loadParcelas);
  // selectedParcelaId guarda el ID real de la parcela
  const [selectedParcelaId, setSelectedParcelaId] = useState(null);
  const [parcelaOpen, setParcelaOpen] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [editando, setEditando] = useState(null);
  const [newActivity, setNewActivity] = useState('');
  const [newTipo, setNewTipo] = useState('Riego');
  const [newFecha, setNewFecha] = useState(() => isoLocal(new Date()));
  const [actividades, setActividades] = useState([]);
  const [diagnosticos, setDiagnosticos] = useState([]);
  const [expandedId, setExpandedId] = useState(null);
  const [hoveredDay, setHoveredDay] = useState(null);
  const [confirmDelete, setConfirmDelete] = useState(null);
  const today = new Date();
  const [viewYear, setViewYear] = useState(today.getFullYear());
  const [viewMonth, setViewMonth] = useState(today.getMonth());

  // Parcela actualmente seleccionada (objeto completo)
  const parcelaSeleccionada = parcelas.find(p => p.id === selectedParcelaId) || parcelas[0];
  const parcelaIdActual = parcelaSeleccionada?.id || null;

  // Recargar parcelas del localStorage cada vez que se monta o se recibe evento
  const recargarParcelas = () => {
    setParcelas(loadParcelas());
  };

  useEffect(() => {
    recargarParcelas();
    // 'storage' funciona entre pestañas; 'parcelasActualizadas' funciona dentro de la misma app
    window.addEventListener('storage', recargarParcelas);
    window.addEventListener('parcelasActualizadas', recargarParcelas);
    return () => {
      window.removeEventListener('storage', recargarParcelas);
      window.removeEventListener('parcelasActualizadas', recargarParcelas);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/exhaustive-deps
    cargarTodo();
  }, [parcelaIdActual]);

  // Recargar cuando se guarda un diagnóstico nuevo desde Diagnóstico
  useEffect(() => {
    const handler = () => cargarTodo();
    window.addEventListener('diagGuardado', handler);
    return () => window.removeEventListener('diagGuardado', handler);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parcelaIdActual]);

  const cargarTodo = () => {
    const key = `historial_${parcelaIdActual}`;
    const saved = localStorage.getItem(key);
    const base = [
      { id: 1, titulo: 'Riego (ejemplo)', fecha: haceDias(2), color: TIPO_COLORES['Riego'], tipo: 'Riego', emoji: '💧' },
      { id: 2, titulo: 'Aporque (ejemplo)', fecha: haceDias(9), color: TIPO_COLORES['Manual'], tipo: 'Manual', emoji: '📋' },
    ];
    const manual = saved ? [...JSON.parse(saved), ...base] : base;  // las de ejemplo no se pueden borrar
    setActividades(manual);

    // Filtrar diagnósticos por el ID real de la parcela
    const diags = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith('diag_')) {
        try {
          const d = JSON.parse(localStorage.getItem(k));
          // Soportar tanto parcelaId (nuevo) como parcela antiguo (índice)
          // Si tiene parcelaId lo usamos, si no intentamos compatibilidad
          const matchById = d.parcelaId !== undefined && d.parcelaId === parcelaIdActual;
          const matchByLegacyId = d.parcela !== undefined && d.parcela === parcelaIdActual;
          if (matchById || matchByLegacyId) diags.push(d);
        } catch(e) {}
      }
    }
    setDiagnosticos(diags.sort((a, b) => b.timestamp - a.timestamp));
  };

  const getDayEvents = (day) => {
    const dateStr = `${viewYear}-${String(viewMonth+1).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
    const acts = actividades.filter(a => a.fecha && a.fecha.startsWith(dateStr));
    const diags = diagnosticos.filter(d => {
      if (!d.timestamp) return false;
      const dd = new Date(d.timestamp);
      return dd.getFullYear() === viewYear && dd.getMonth() === viewMonth && dd.getDate() === day;
    });
    return { acts, diags };
  };

  const abrirNueva = () => {
    setEditando(null);
    setNewActivity('');
    setNewTipo('Riego');
    setNewFecha(isoLocal(new Date()));
    setShowModal(true);
  };

  const abrirEditar = (act) => {
    setEditando(act);
    setNewActivity(act.titulo);
    setNewTipo(act.tipo || 'Manual');
    setNewFecha(act.fecha || isoLocal(new Date()));
    setShowModal(true);
  };

  const guardarActividad = () => {
    const tipo = newTipo || 'Manual';
    const key = `historial_${parcelaIdActual}`;
    const saved = localStorage.getItem(key);
    let existing = saved ? JSON.parse(saved) : [];

    if (editando) {
      existing = existing.map(a => a.id === editando.id ? {
        ...a,
        titulo: newActivity,
        tipo,
        fecha: newFecha,
        color: TIPO_COLORES[tipo] || '#888',
        emoji: tipo === 'Riego' ? '💧' : tipo === 'Fertilización' ? '🌿' : tipo === 'Fumigación' ? '🧴' : tipo === 'Cosecha' ? '🌾' : '📋'
      } : a);
    } else {
      const nueva = {
        id: Date.now(),
        titulo: newActivity,
        fecha: newFecha,
        color: TIPO_COLORES[tipo] || '#888',
        tipo,
        emoji: tipo === 'Riego' ? '💧' : tipo === 'Fertilización' ? '🌿' : tipo === 'Fumigación' ? '🧴' : tipo === 'Cosecha' ? '🌾' : '📋'
      };
      existing = [nueva, ...existing];
    }

    localStorage.setItem(key, JSON.stringify(existing));
    cargarTodo();
    setShowModal(false);
    setEditando(null);
    setNewActivity('');
  };

  const eliminarDiag = (id) => {
    localStorage.removeItem(`diag_${id}`);
    cargarTodo();
    setExpandedId(null);
    setConfirmDelete(null);
  };

  const eliminarActividad = (id) => {
    const key = `historial_${parcelaIdActual}`;
    const saved = localStorage.getItem(key);
    if (!saved) return;
    const existing = JSON.parse(saved).filter(a => a.id !== id);
    localStorage.setItem(key, JSON.stringify(existing));
    cargarTodo();
    setConfirmDelete(null);
  };

  const prevMonth = () => {
    if (viewMonth === 0) { setViewMonth(11); setViewYear(y => y - 1); }
    else setViewMonth(m => m - 1);
  };

  const nextMonth = () => {
    if (viewMonth === 11) { setViewMonth(0); setViewYear(y => y + 1); }
    else setViewMonth(m => m + 1);
  };

  const daysInMonth = getDaysInMonth(viewYear, viewMonth);
  const firstDay = getFirstDayOfMonth(viewYear, viewMonth);
  const isToday = (day) => day === today.getDate() && viewMonth === today.getMonth() && viewYear === today.getFullYear();

  const allActividades = [
    ...actividades.map(a => ({ ...a, displayTitle: `${a.emoji || '📋'} ${a.titulo}`, displayDate: a.fecha })),
    ...diagnosticos.map(d => ({
      id: d.id, titulo: d.resultado?.enfermedad, fecha: isoLocal(new Date(d.timestamp)),
      color: d.resultado?.sana ? '#1D9E75' : '#E24B4A', tipo: 'IA', emoji: '🔍',
      displayTitle: `🔍 ${d.resultado?.enfermedad}`,
      displayDate: d.fecha, isDiag: true, diagData: d
    }))
  ].sort((a, b) => (b.fecha || '') > (a.fecha || '') ? 1 : -1);

  return (
    <>
      <div className="header">
        <div className="logo">
          <img src="/AGROTEC.png" alt="AGROTEC" style={{width:32,height:32,borderRadius:8,objectFit:"contain"}} />
          Historial
        </div>
        <div style={{ fontSize: 20, cursor: 'pointer' }} onClick={abrirNueva}>➕</div>
      </div>

      <div className="page">
        {/* Dropdown de parcelas dinámico desde localStorage */}
        <div className="dropdown" style={{ marginBottom: 14 }}>
          <div className="dropdown-header" onClick={() => setParcelaOpen(!parcelaOpen)}>
            <span>
              {parcelaSeleccionada?.emoji} {parcelaSeleccionada?.nombre}
              {parcelaSeleccionada?.zona ? ` — ${parcelaSeleccionada.zona}` : ''}
            </span>
            <span>{parcelaOpen ? '▲' : '▼'}</span>
          </div>
          {parcelaOpen && parcelas.map((p) => (
            <div key={p.id} className={`dropdown-option ${p.id === parcelaIdActual ? 'selected' : ''}`}
              onClick={() => { setSelectedParcelaId(p.id); setParcelaOpen(false); }}>
              {p.emoji} {p.nombre}{p.zona ? ` — ${p.zona}` : ''}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <button onClick={prevMonth} style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'white', borderRadius: 8, padding: '4px 12px', cursor: 'pointer', fontSize: 16 }}>‹</button>
          <div style={{ fontSize: 14, fontWeight: 600 }}>{MESES[viewMonth]} {viewYear}</div>
          <button onClick={nextMonth} style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'white', borderRadius: 8, padding: '4px 12px', cursor: 'pointer', fontSize: 16 }}>›</button>
        </div>

        <div className="cal-grid">
          {['L','M','X','J','V','S','D'].map(d => <div key={d} className="cal-day hdr">{d}</div>)}
          {Array(firstDay).fill(null).map((_, i) => <div key={`e${i}`} className="cal-day"></div>)}
          {Array(daysInMonth).fill(null).map((_, i) => {
            const day = i + 1;
            const { acts, diags } = getDayEvents(day);
            const hasActs = acts.length > 0;
            const hasDiags = diags.length > 0;
            const total = acts.length + diags.length;
            return (
              <div key={day} style={{ position: 'relative' }}
                onMouseEnter={() => total > 0 && setHoveredDay(day)}
                onMouseLeave={() => setHoveredDay(null)}>
                <div className={`cal-day ${isToday(day) ? 'td' : hasDiags ? 'al' : hasActs ? 'ev' : ''}`}
                  style={{ cursor: total > 0 ? 'pointer' : 'default', position: 'relative' }}>
                  {day}
                  {total > 0 && (
                    <div style={{ position: 'absolute', bottom: 1, left: '50%', transform: 'translateX(-50%)', display: 'flex', gap: 1 }}>
                      {hasActs && <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#1D9E75' }}></div>}
                      {hasDiags && <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#E24B4A' }}></div>}
                    </div>
                  )}
                </div>
                {hoveredDay === day && total > 0 && (
                  <div style={{ position: 'absolute', bottom: '110%', left: '50%', transform: 'translateX(-50%)', background: 'var(--bg-card2)', border: '1px solid var(--border)', borderRadius: 8, padding: '6px 10px', zIndex: 100, minWidth: 150, fontSize: 10, whiteSpace: 'nowrap' }}>
                    {acts.map((a, i) => <div key={i} style={{ color: a.color, marginBottom: 2 }}>• {a.titulo}</div>)}
                    {diags.map((d, i) => <div key={i} style={{ color: '#E24B4A', marginBottom: 2 }}>• {d.resultado?.enfermedad}</div>)}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div style={{ display: 'flex', gap: 12, marginBottom: 14 }}>
          <span style={{ fontSize: 9, color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center', gap: 3 }}>
            <div style={{ width: 8, height: 8, background: 'rgba(29,158,117,0.2)', border: '1px solid var(--green)', borderRadius: 2 }}></div>Actividad
          </span>
          <span style={{ fontSize: 9, color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center', gap: 3 }}>
            <div style={{ width: 8, height: 8, background: 'rgba(226,75,74,0.2)', border: '1px solid var(--red)', borderRadius: 2 }}></div>Diagnóstico IA
          </span>
        </div>

        <div className="divider" />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div className="section-label" style={{ marginBottom: 0 }}>📋 Todo el historial</div>
          <button onClick={abrirNueva}
            style={{ background: 'rgba(29,158,117,0.15)', border: '1px solid rgba(29,158,117,0.3)', color: 'var(--green)', borderRadius: 8, padding: '5px 10px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600 }}>
            + Agregar
          </button>
        </div>

        {allActividades.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-tertiary)', fontSize: 12, padding: '20px 0' }}>
            No hay actividades aún
          </div>
        )}

        {allActividades.map(a => (
          <div key={a.id}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div className="timeline-item" style={{ cursor: 'pointer', flex: 1, marginBottom: 0 }}
                onClick={() => setExpandedId(expandedId === a.id ? null : a.id)}>
                <div className="tl-dot" style={{ background: a.color }}></div>
                <div style={{ flex: 1 }}>
                  <div className="tl-title">{a.displayTitle}</div>
                  <div className="tl-date">{a.displayDate} · {a.tipo}</div>
                </div>
                {a.isDiag && a.diagData?.foto && expandedId !== a.id && (
                  <img src={a.diagData.foto} alt=""
                    style={{ width: 40, height: 40, borderRadius: 8, objectFit: 'cover', border: '1px solid var(--border)', flexShrink: 0, marginLeft: 6 }} />
                )}
                <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginLeft: 4 }}>
                  {expandedId === a.id ? '▲' : '▼'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 4, flexShrink: 0 }}>
                {!a.isDiag && (
                  <button onClick={() => abrirEditar(a)}
                    style={{ background: 'rgba(55,138,221,0.15)', border: '1px solid rgba(55,138,221,0.3)', color: 'var(--blue)', borderRadius: 6, padding: '4px 8px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit' }}>
                    ✏️
                  </button>
                )}
                <button onClick={() => setConfirmDelete({ id: a.id, tipo: a.isDiag ? 'diag' : 'act', nombre: a.displayTitle })}
                  style={{ background: 'rgba(226,75,74,0.15)', border: '1px solid rgba(226,75,74,0.3)', color: 'var(--red)', borderRadius: 6, padding: '4px 8px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit' }}>
                  🗑️
                </button>
              </div>
            </div>

            {expandedId === a.id && (
              <div style={{ marginLeft: 22, background: 'var(--bg-card)', borderRadius: 10, padding: 12, marginBottom: 10, border: '1px solid var(--border)' }}>
                {a.isDiag ? (
                  <>
                    {a.diagData?.foto
                      ? <img src={a.diagData.foto} alt="diagnóstico" style={{ width: '100%', borderRadius: 8, maxHeight: 180, objectFit: 'cover', marginBottom: 8 }} />
                      : <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginBottom: 8, textAlign: 'center' }}>📷 Sin foto guardada</div>
                    }
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6 }}>{a.diagData?.resultado?.descripcion}</div>
                    {a.diagData?.resultado?.confianza !== undefined && (
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6 }}>Confianza del modelo: <strong style={{ color: 'white' }}>{a.diagData.resultado.confianza}%</strong></div>
                    )}
                    {a.diagData?.resultado?.recomendacion && (
                      <>
                        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--blue)', marginBottom: 4 }}>🌱 ¿Qué hacer?</div>
                        <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{a.diagData.resultado.recomendacion}</div>
                      </>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    <div style={{ marginBottom: 4 }}>📅 Fecha: <strong style={{ color: 'white' }}>{a.fecha}</strong></div>
                    <div>🏷️ Tipo: <strong style={{ color: 'white' }}>{a.tipo}</strong></div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {showModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 2000, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
          <div style={{ background: 'var(--bg-secondary)', borderRadius: '20px 20px 0 0', padding: 24, width: '100%', maxWidth: 430 }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              {editando ? '✏️ Editar actividad' : '➕ Nueva actividad'}
            </div>
            <label className="input-label">Descripción</label>
            <input className="input-field" placeholder="Ej: Aporque, riego, siembra..." value={newActivity} onChange={e => setNewActivity(e.target.value)} />
            <div className="grid-2">
              <div>
                <label className="input-label">Tipo</label>
                <select className="input-field" value={newTipo} onChange={e => setNewTipo(e.target.value)}>
                  <option>Riego</option>
                  <option>Fertilización</option>
                  <option>Fumigación</option>
                  <option>Cosecha</option>
                  <option>Siembra</option>
                  <option>Manual</option>
                </select>
              </div>
              <div>
                <label className="input-label">Fecha</label>
                <input className="input-field" type="date" value={newFecha} onChange={e => setNewFecha(e.target.value)} />
              </div>
            </div>
            <button className="btn-primary" onClick={guardarActividad} disabled={!newActivity}>
              {editando ? 'Guardar cambios' : 'Guardar actividad'}
            </button>
            <button onClick={() => { setShowModal(false); setEditando(null); }}
              style={{ width: '100%', padding: 12, background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: 13, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit' }}>
              Cancelar
            </button>
          </div>
        </div>
      )}

      {confirmDelete && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 3000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
          <div style={{ background: 'var(--bg-secondary)', borderRadius: 16, padding: 24, width: '100%', maxWidth: 340, textAlign: 'center' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>🗑️</div>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>¿Eliminar registro?</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 20 }}>
              "{confirmDelete.nombre}"<br/>Esta acción no se puede deshacer.
            </div>
            <div style={{ display: 'flex', gap: 10 }}>
              <button onClick={() => setConfirmDelete(null)}
                style={{ flex: 1, padding: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'white', borderRadius: 10, cursor: 'pointer', fontFamily: 'inherit', fontSize: 13 }}>
                Cancelar
              </button>
              <button onClick={() => confirmDelete.tipo === 'diag' ? eliminarDiag(confirmDelete.id) : eliminarActividad(confirmDelete.id)}
                style={{ flex: 1, padding: 12, background: 'rgba(226,75,74,0.2)', border: '1px solid var(--red)', color: 'var(--red)', borderRadius: 10, cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 600 }}>
                Eliminar
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
