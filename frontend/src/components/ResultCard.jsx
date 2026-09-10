import React from 'react';
import { apiUrl } from '../api.js';

export default function ResultCard({
  result,
  isSelected,
  onSelect,
  onFindSimilar,
  isSimilar = false
}) {
  const badgeClass = isSimilar ? 'badge-similar' : (
    result.change_confidence < 0.40 ? 'badge-suppressed' :
      (result.change_confidence < 0.70 ? 'badge-review' : 'badge-high')
  );

  return (
    <div
      onClick={onSelect}
      style={{
        background: isSelected ? '#1e293b' : '#0f172a',
        border: isSelected ? '1px solid #22d3ee' : '1px solid #334155',
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
        {isSimilar && result.thumbnail_path && (
          <img
            src={apiUrl(`/static/tiles/${result.thumbnail_path.split(/[\\/]/).pop()}`)}
            alt="Similar site thumbnail"
            style={{ width: '42px', height: '42px', objectFit: 'cover', borderRadius: '4px', marginRight: '8px' }}
          />
        )}
        <div>
          <span style={{ fontWeight: 600, fontSize: '13px', color: '#f8fafc' }}>
            {result.location_key}
          </span>
          {result.coordinates && (
            <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px' }}>
              {result.coordinates[1]?.toFixed(4)}°N, {result.coordinates[0]?.toFixed(4)}°E
            </div>
          )}
        </div>
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
          {isSimilar ? 'similar site' : result.change_type.replace('_', ' ')}
        </span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
        <span>{isSimilar ? 'Semantic Similarity:' : 'Change Confidence:'}</span>
        <span style={{ fontWeight: 600, color: '#f8fafc' }}>{isSimilar ? result.similarity : result.change_confidence}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
        <span>{isSimilar ? 'Acquisition:' : 'Earliest Change:'}</span>
        <span style={{ color: '#cbd5e1' }}>{isSimilar ? (result.acquisition_dates?.[0] || 'N/A') : (result.earliest_change_date || 'N/A')}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
        <span>Sensor: {result.sensor}</span>
        <span>Sim: {result.similarity}</span>
      </div>

      {!isSimilar && onFindSimilar && (
        <button
          type="button"
          onClick={(event) => { event.stopPropagation(); onFindSimilar(result); }}
          style={{ padding: '6px 8px', background: '#12232b', border: '1px solid #155e75', borderRadius: '4px', color: '#67e8f9', cursor: 'pointer', fontSize: '11px' }}
        >
          Find similar sites
        </button>
      )}
    </div>
  );
}
