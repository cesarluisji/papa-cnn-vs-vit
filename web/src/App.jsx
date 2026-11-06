import React, { useState, useEffect } from 'react';
import './styles/global.css';
import Parcelas from './pages/Parcelas';
import Diagnostico from './pages/Diagnostico';
import Historial from './pages/Historial';
import { cargarModelo } from './modelo';

export default function App() {
  const [activePage, setActivePage] = useState('parcelas');

  // El modelo (≈24 MB) empieza a descargarse apenas se abre la app, en cualquier pestaña
  useEffect(() => { cargarModelo().catch(() => {}); }, []);

  useEffect(() => {
    const handler = () => setActivePage('historial');
    window.addEventListener('goToHistorial', handler);
    return () => window.removeEventListener('goToHistorial', handler);
  }, []);

  return (
    <div className="app">
      {/* Todas las páginas montadas, solo se ocultan con display */}
      <div style={{ display: activePage === 'parcelas' ? 'block' : 'none' }}><Parcelas visible={activePage === 'parcelas'} /></div>
      <div style={{ display: activePage === 'diagnostico' ? 'block' : 'none' }}><Diagnostico /></div>
      <div style={{ display: activePage === 'historial' ? 'block' : 'none' }}><Historial /></div>

      <nav className="bottom-nav">
        <div className={`nav-item ${activePage === 'parcelas' ? 'active' : ''}`} onClick={() => setActivePage('parcelas')}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
          Parcelas
        </div>
        <div className={`nav-item ${activePage === 'diagnostico' ? 'active' : ''}`} onClick={() => setActivePage('diagnostico')}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Diagnóstico
        </div>
        <div className={`nav-item ${activePage === 'historial' ? 'active' : ''}`} onClick={() => setActivePage('historial')}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
          Historial
        </div>
      </nav>
    </div>
  );
}
