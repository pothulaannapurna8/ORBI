import React, { useState } from 'react';
import { apiFetch } from '../api.js';

export default function ReviewPanel({ result, onReviewSubmit, onFindSimilar, similarLoading }) {
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [lastDecision, setLastDecision] = useState(null);
  const [reviewError, setReviewError] = useState('');

  if (!result) return null;

  // Extract CCS breakdown components
  const ccsBreakdown = result.ccs_breakdown || {
    change_evidence: result.change_confidence * 0.4,
    cloud_score: 0.1,
    registration_quality: 0.9,
    temporal_consistency: 0.8
  };

  const handleReview = async (decision) => {
    setSubmitting(true);
    setReviewError('');
    if (!note.trim()) {
      setReviewError('Add a one-line analyst note before recording the decision.');
      setSubmitting(false);
      return;
    }
    try {
      const res = await apiFetch(`/review/${result.tile_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision,
          analyst_note: note
        })
      });
      if (res.ok) {
        setLastDecision(decision);
        setNote('');
        if (onReviewSubmit) onReviewSubmit(result.tile_id, decision);
      } else {
        setReviewError(`Review could not be recorded (${res.status}).`);
      }
    } catch (err) {
      console.error(err);
      setReviewError('Review service unavailable.');
    } finally {
      setSubmitting(false);
    }
  };

  const getComponentColor = (value) => {
    if (value >= 0.7) return '#4ade80';
    if (value >= 0.4) return '#fbbf24';
    return '#f87171';
  };

  return (
    <div style={{
      background: '#1e293b',
      borderRadius: '8px',
      border: '1px solid #334155',
      padding: '14px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ fontSize: '13px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>
          Analyst Verification Workflow
        </h4>
        {lastDecision && (
          <span style={{
            fontSize: '11px',
            fontWeight: 600,
            color: lastDecision === 'confirmed' ? '#4ade80' : '#f87171'
          }}>
            Recorded: {lastDecision.toUpperCase()}
          </span>
        )}
      </div>

      {/* CCS Component Breakdown */}
      <div style={{
        background: '#0f172a',
        borderRadius: '6px',
        padding: '10px',
        border: '1px solid #334155'
      }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>
          Change Confidence Score Breakdown
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '11px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#cbd5e1' }}>Change Evidence (40%)</span>
            <span style={{ color: getComponentColor(ccsBreakdown.change_evidence), fontWeight: 600 }}>
              {ccsBreakdown.change_evidence.toFixed(3)}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#cbd5e1' }}>Cloud Quality (20%)</span>
            <span style={{ color: getComponentColor(1 - ccsBreakdown.cloud_score), fontWeight: 600 }}>
              {(1 - ccsBreakdown.cloud_score).toFixed(3)}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#cbd5e1' }}>Registration (20%)</span>
            <span style={{ color: getComponentColor(ccsBreakdown.registration_quality), fontWeight: 600 }}>
              {ccsBreakdown.registration_quality.toFixed(3)}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#cbd5e1' }}>Temporal (20%)</span>
            <span style={{ color: getComponentColor(ccsBreakdown.temporal_consistency), fontWeight: 600 }}>
              {ccsBreakdown.temporal_consistency.toFixed(3)}
            </span>
          </div>
        </div>
        
        {/* Overall CCS Bar */}
        <div style={{ marginTop: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '11px' }}>
            <span style={{ color: '#94a3b8' }}>Overall CCS</span>
            <span style={{ color: getComponentColor(result.change_confidence), fontWeight: 700 }}>
              {result.change_confidence.toFixed(3)}
            </span>
          </div>
          <div style={{
            width: '100%',
            height: '6px',
            background: '#334155',
            borderRadius: '3px',
            overflow: 'hidden'
          }}>
            <div
              style={{
                width: `${result.change_confidence * 100}%`,
                height: '100%',
                background: getComponentColor(result.change_confidence),
                transition: 'width 0.3s ease'
              }}
            />
          </div>
        </div>
      </div>

      {/* Sensor Metadata */}
      <div style={{
        background: '#0f172a',
        borderRadius: '6px',
        padding: '10px',
        border: '1px solid #334155',
        fontSize: '11px'
      }}>
        <div style={{ fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase' }}>
          Sensor Metadata
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span style={{ color: '#cbd5e1' }}>Platform:</span>
          <span style={{ color: '#f8fafc', fontWeight: 500 }}>{result.sensor || 'Sentinel-2 L2A'}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: '#cbd5e1' }}>Resolution:</span>
          <span style={{ color: '#f8fafc', fontWeight: 500 }}>10m GSD</span>
        </div>
      </div>

      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Enter analyst justification notes (e.g. 'Verified foundation trenches excavated')..."
        style={{
          width: '100%',
          height: '55px',
          background: '#0f172a',
          border: '1px solid #475569',
          borderRadius: '6px',
          color: '#f8fafc',
          padding: '8px',
          fontSize: '12px',
          resize: 'none',
          outline: 'none'
        }}
      />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        <button
          onClick={() => handleReview('confirmed')}
          disabled={submitting || !note.trim()}
          style={{
            padding: '8px',
            borderRadius: '6px',
            border: 'none',
            background: '#16a34a',
            color: 'white',
            fontWeight: 600,
            fontSize: '12px',
            cursor: submitting ? 'not-allowed' : 'pointer'
          }}
        >
          ✓ Confirm Change
        </button>

        <button
          onClick={() => handleReview('rejected')}
          disabled={submitting || !note.trim()}
          style={{
            padding: '8px',
            borderRadius: '6px',
            border: 'none',
            background: '#dc2626',
            color: 'white',
            fontWeight: 600,
            fontSize: '12px',
            cursor: submitting ? 'not-allowed' : 'pointer'
          }}
        >
          ✗ Reject (False Alarm)
        </button>
      </div>

      {reviewError && <div style={{ color: '#fbbf24', fontSize: '11px' }}>{reviewError}</div>}
      {lastDecision && <div className="review-toast">Review recorded: {lastDecision}</div>}

      {onFindSimilar && (
        <button
          type="button"
          onClick={() => onFindSimilar(result)}
          disabled={similarLoading}
          style={{ padding: '8px', borderRadius: '6px', border: '1px solid #155e75', background: '#12232b', color: '#67e8f9', fontWeight: 600, fontSize: '12px', cursor: similarLoading ? 'wait' : 'pointer' }}
        >
          {similarLoading ? 'Finding similar sites...' : 'Find similar sites'}
        </button>
      )}
    </div>
  );
}
