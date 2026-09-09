import React, { useState, useRef, useEffect } from 'react';
import { apiUrl } from '../api.js';

export default function BeforeAfterViewer({ result }) {
  const [showMask, setShowMask] = useState(true);
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'overlay' | 'sidebyside'
  const [sliderPos, setSliderPos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  if (!result) {
    return (
      <div style={{ color: '#64748b', textAlign: 'center', padding: '40px 20px', fontSize: '14px' }}>
        Select a detection candidate from the map or ranked list to inspect before/after imagery.
      </div>
    );
  }

  const beforeUrl = result.evidence_paths?.before_png
    ? apiUrl(`/static/tiles/${result.evidence_paths.before_png.split(/[\\/]/).pop()}`)
    : null;

  const afterUrl = result.evidence_paths?.after_png
    ? apiUrl(`/static/tiles/${result.evidence_paths.after_png.split(/[\\/]/).pop()}`)
    : null;

  const handleMouseDown = (e) => {
    setIsDragging(true);
    e.preventDefault();
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percentage);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ fontSize: '13px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>
          Multi-Temporal Evidence Viewer
        </h4>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={() => setViewMode('split')}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #475569',
              background: viewMode === 'split' ? '#0284c7' : '#1e293b',
              color: '#f8fafc',
              fontSize: '11px',
              cursor: 'pointer'
            }}
          >
            Split Slider
          </button>
          <button
            onClick={() => setViewMode('sidebyside')}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #475569',
              background: viewMode === 'sidebyside' ? '#0284c7' : '#1e293b',
              color: '#f8fafc',
              fontSize: '11px',
              cursor: 'pointer'
            }}
          >
            Side-by-Side
          </button>
          <button
            onClick={() => setShowMask(!showMask)}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #475569',
              background: showMask ? '#0284c7' : '#1e293b',
              color: '#f8fafc',
              fontSize: '11px',
              cursor: 'pointer'
            }}
          >
            {showMask ? 'Hide Mask' : 'Show Mask'}
          </button>
        </div>
      </div>

      {/* Interactive Split-Screen Slider View */}
      {viewMode === 'split' && (
        <div 
          ref={containerRef}
          style={{
            position: 'relative',
            width: '100%',
            height: '200px',
            background: '#0f172a',
            borderRadius: '6px',
            overflow: 'hidden',
            border: '1px solid #334155',
            cursor: 'col-resize'
          }}
          onMouseDown={handleMouseDown}
        >
          {/* Before Image (Background) */}
          {beforeUrl && (
            <img
              src={beforeUrl}
              alt="Before"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'cover'
              }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          )}

          {/* After Image (Clipped with slider) */}
          {afterUrl && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: `${sliderPos}%`,
                height: '100%',
                overflow: 'hidden',
                borderRight: '2px solid #0284c7'
              }}
            >
              <img
                src={afterUrl}
                alt="After"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '100%',
                  width: containerRef.current ? `${containerRef.current.offsetWidth}px` : '100%',
                  objectFit: 'cover'
                }}
                onError={(e) => { e.target.style.display = 'none'; }}
              />
              
              {/* Change Mask Overlay on After Image */}
              {showMask && result.change_type === 'construction' && (
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  background: 'rgba(239, 68, 68, 0.25)',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fca5a5',
                  fontSize: '11px',
                  fontWeight: 600
                }}>
                  Open-CD Change
                </div>
              )}

              {showMask && result.change_type.includes('water') && (
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  background: 'rgba(56, 189, 248, 0.25)',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#bae6fd',
                  fontSize: '11px',
                  fontWeight: 600
                }}>
                  NDWI Water
                </div>
              )}
            </div>
          )}

          {/* Slider Handle */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: `${sliderPos}%`,
              transform: 'translateX(-50%)',
              width: '4px',
              height: '100%',
              background: '#0284c7',
              cursor: 'col-resize',
              zIndex: 10
            }}
          >
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '20px',
              height: '20px',
              background: '#0284c7',
              borderRadius: '50%',
              border: '2px solid white',
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
            }} />
          </div>

          {/* Date Labels */}
          <div style={{
            position: 'absolute',
            bottom: '8px',
            left: '8px',
            background: 'rgba(15, 23, 42, 0.8)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            color: '#cbd5e1',
            fontWeight: 500
          }}>
            T1 (Before): {result.acquisition_dates?.[0] || '2024-03-01'}
          </div>
          <div style={{
            position: 'absolute',
            bottom: '8px',
            right: '8px',
            background: 'rgba(15, 23, 42, 0.8)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            color: '#cbd5e1',
            fontWeight: 500
          }}>
            T2 (After): {result.acquisition_dates?.[1] || '2025-11-20'}
          </div>
        </div>
      )}

      {/* Side-by-Side View */}
      {viewMode === 'sidebyside' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          {/* Before Tile */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
              <span>T1 (Before)</span>
              <span style={{ color: '#cbd5e1' }}>{result.acquisition_dates?.[0] || '2024-03-01'}</span>
            </div>
            <div style={{
              position: 'relative',
              width: '100%',
              height: '180px',
              background: '#0f172a',
              borderRadius: '6px',
              overflow: 'hidden',
              border: '1px solid #334155'
            }}>
              {beforeUrl ? (
                <img
                  src={beforeUrl}
                  alt="Before"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#475569' }}>
                  Preview Staged
                </div>
              )}
            </div>
          </div>

          {/* After Tile */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
              <span>T2 (After)</span>
              <span style={{ color: '#cbd5e1' }}>{result.acquisition_dates?.[1] || '2025-11-20'}</span>
            </div>
            <div style={{
              position: 'relative',
              width: '100%',
              height: '180px',
              background: '#0f172a',
              borderRadius: '6px',
              overflow: 'hidden',
              border: '1px solid #334155'
            }}>
              {afterUrl ? (
                <img
                  src={afterUrl}
                  alt="After"
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#475569' }}>
                  Preview Staged
                </div>
              )}

              {/* Change Mask Highlight Overlay */}
              {showMask && result.change_type === 'construction' && (
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  background: 'rgba(239, 68, 68, 0.25)',
                  border: '2px dashed #ef4444',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fca5a5',
                  fontSize: '11px',
                  fontWeight: 600
                }}>
                  Open-CD Change Boundary
                </div>
              )}

              {showMask && result.change_type.includes('water') && (
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  background: 'rgba(56, 189, 248, 0.25)',
                  border: '2px dashed #38bdf8',
                  pointerEvents: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#bae6fd',
                  fontSize: '11px',
                  fontWeight: 600
                }}>
                  NDWI Water Boundary
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
