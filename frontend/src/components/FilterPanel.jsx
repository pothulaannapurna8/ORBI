import React from 'react';

export default function FilterPanel({
  aoi,
  setAoi,
  sensor,
  setSensor,
  topK,
  setTopK,
  showSuppressed,
  setShowSuppressed
}) {
  return (
    <div style={{
      width: '240px',
      background: '#1e293b',
      borderRight: '1px solid #334155',
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '18px',
      fontSize: '13px'
    }}>
      <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Filter Controls
      </h3>

      {/* Preset AOI Selector */}
      <div>
        <label style={{ display: 'block', marginBottom: '6px', color: '#cbd5e1', fontWeight: 500 }}>
          Area of Interest (AOI)
        </label>
        <select
          value={aoi}
          onChange={(e) => setAoi(e.target.value)}
          style={{
            width: '100%',
            padding: '8px',
            borderRadius: '6px',
            border: '1px solid #475569',
            background: '#0f172a',
            color: '#f8fafc',
            outline: 'none'
          }}
        >
          <option value="ALL">All Staged AOIs</option>
          <option value="URBAN">Bangalore Urban Growth (12.97°N, 77.59°E)</option>
          <option value="RIVER">Chennai River Basin (13.08°N, 80.27°E)</option>
        </select>
      </div>

      {/* Sensor Platform */}
      <div>
        <label style={{ display: 'block', marginBottom: '6px', color: '#cbd5e1', fontWeight: 500 }}>
          Sensor Platform
        </label>
        <select
          value={sensor}
          onChange={(e) => setSensor(e.target.value)}
          style={{
            width: '100%',
            padding: '8px',
            borderRadius: '6px',
            border: '1px solid #475569',
            background: '#0f172a',
            color: '#f8fafc',
            outline: 'none'
          }}
        >
          <option value="Sentinel-2 L2A">Sentinel-2 L2A (10m GSD)</option>
          <option value="Landsat-8">Landsat-8 OLI (30m GSD)</option>
        </select>
      </div>

      {/* Top-K Slider */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', color: '#cbd5e1' }}>
          <span>Max Results (Top-K)</span>
          <span style={{ fontWeight: 600, color: '#0284c7' }}>{topK}</span>
        </div>
        <input
          type="range"
          min="5"
          max="50"
          step="5"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
          style={{ width: '100%', accentColor: '#0284c7' }}
        />
      </div>

      {/* Suppressed / False Alarm Toggle */}
      <div style={{
        marginTop: 'auto',
        background: '#0f172a',
        padding: '12px',
        borderRadius: '6px',
        border: '1px solid #334155'
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showSuppressed}
            onChange={(e) => setShowSuppressed(e.target.checked)}
            style={{ accentColor: '#0284c7' }}
          />
          <span style={{ fontSize: '12px', color: '#cbd5e1' }}>
            Show Suppressed False Alarms (CCS &lt; 0.40)
          </span>
        </label>
        <p style={{ fontSize: '11px', color: '#64748b', marginTop: '6px' }}>
          Auditing suppressed items prevents missed change candidates.
        </p>
      </div>
    </div>
  );
}
