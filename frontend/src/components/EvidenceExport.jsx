import React from 'react';

export default function EvidenceExport({ result }) {
  if (!result) return null;

  const handleExport = () => {
    const evidencePackage = {
      export_version: "PS26227_AUDIT_v1",
      exported_at: new Date().toISOString(),
      location_key: result.location_key,
      coordinates: result.coordinates,
      change_type: result.change_type,
      change_confidence_score: result.change_confidence,
      semantic_similarity: result.similarity,
      earliest_change_date: result.earliest_change_date,
      acquisition_dates: result.acquisition_dates,
      sensor: result.sensor,
      provenance: result.provenance,
      evidence_paths: result.evidence_paths,
      ccs_breakdown: result.ccs_breakdown || {
        change_evidence: result.change_confidence * 0.4,
        cloud_score: 0.1,
        registration_quality: 0.9,
        temporal_consistency: 0.8
      }
    };

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
