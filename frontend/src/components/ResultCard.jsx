import React from 'react';

export default function ResultCard({
  result,
  isSelected,
  onSelect
}) {
  let badgeClass = 'badge-high';
  if (result.change_confidence < 0.40) badgeClass = 'badge-suppressed';
  else if (result.change_confidence < 0.70) badgeClass = 'badge-review';

  return (
    <div
      onClick={onSelect}
      style={{
        background: isSelected ? '#1e293b' : '#0f172a',
        border: isSelected ? '1px solid #0284c7' : '1px solid #334155',
        borderRadius: '8px',
        padding: '12px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        transition: 'all 0.15s ease'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 600, fontSize: '13px', color: '#f8fafc' }}>
          {result.location_key}
        </span>
        <span
          className={badgeClass}
          style={{
            fontSize: '11px',
            fontWeight: 600,
            padding: '2px 6px',
            borderRadius: '4px',
            textTransform: 'uppercase'
          }}
        >
          {result.change_type.replace('_', ' ')}
        </span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
        <span>Change Confidence:</span>
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>{result.change_confidence}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
        <span>Earliest Change:</span>
        <span style={{ color: '#cbd5e1' }}>{result.earliest_change_date || 'N/A'}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
        <span>Sensor: {result.sensor}</span>
        <span>Sim: {result.similarity}</span>
      </div>
    </div>
  );
}
