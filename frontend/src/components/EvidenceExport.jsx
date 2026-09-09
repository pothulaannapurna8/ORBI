import React from 'react';
import { apiFetch } from '../api.js';

export default function EvidenceExport({ result }) {
  if (!result) return null;

  const handleExport = async () => {
    if (!result.change_id) return;
    const response = await apiFetch(`/results/${result.change_id}/evidence`);
    if (!response.ok) return;
    const evidencePackage = await response.json();
    /* The API response is the audit bundle; the browser only handles download. */
    const blob = new Blob([JSON.stringify(evidencePackage, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `evidence_${result.location_key}_${result.earliest_change_date || 'audit'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };


  return (
    <button
      onClick={handleExport}
      style={{
        width: '100%',
        padding: '8px 12px',
        background: '#1e293b',
        border: '1px solid #475569',
        borderRadius: '6px',
        color: '#93c5fd',
        fontSize: '12px',
        fontWeight: 500,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '6px'
      }}
    >
      📥 Export Evidence Package (.json)
    </button>
  );
}
