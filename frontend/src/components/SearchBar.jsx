import React, { useRef, useState } from 'react';

export default function SearchBar({
  query,
  setQuery,
  imageFile,
  setImageFile,
  onSearch,
  loading,
  useMultimodal,
  setUseMultimodal,
  dateRange,
  setDateRange
}) {
  const fileInputRef = useRef(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  const clearImage = () => {
    setImageFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center', width: '100%' }}>
        <div style={{ position: 'relative', flex: 1, display: 'flex', alignItems: 'center' }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onSearch()}
            placeholder="Enter natural-language query (e.g. 'newly built structures near rivers between 2024 and 2026')..."
            style={{
              width: '100%',
              padding: '10px 16px',
              paddingRight: imageFile ? '140px' : '40px',
              borderRadius: '8px',
              border: '1px solid #334155',
              background: '#0f172a',
              color: '#f8fafc',
              fontSize: '14px',
              outline: 'none'
            }}
          />
          {imageFile && (
            <div style={{
              position: 'absolute',
              right: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: '#1e293b',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              border: '1px solid #0284c7'
            }}>
              <span style={{ maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                📷 {imageFile.name}
              </span>
              <button
                onClick={clearImage}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '14px' }}
              >
                ×
              </button>
            </div>
          )}
        </div>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/png,image/jpeg,image/tiff"
          style={{ display: 'none' }}
        />

        <button
          onClick={() => fileInputRef.current?.click()}
          title="Upload satellite image query (GeoTIFF/PNG/JPEG)"
          style={{
            padding: '10px 14px',
            background: imageFile ? '#0284c7' : '#1e293b',
            border: '1px solid #334155',
            borderRadius: '8px',
            color: '#f8fafc',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '14px'
          }}
        >
          📷 Upload
        </button>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{
            padding: '10px 14px',
            background: showAdvanced ? '#0284c7' : '#1e293b',
            border: '1px solid #334155',
            borderRadius: '8px',
            color: '#f8fafc',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '14px'
          }}
        >
          ⚙️
        </button>

        <button
          onClick={onSearch}
          disabled={loading}
          style={{
            padding: '10px 22px',
            background: loading ? '#0369a1' : '#0284c7',
            border: 'none',
            borderRadius: '8px',
            color: 'white',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '14px'
          }}
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {/* Advanced Options Panel */}
      {showAdvanced && (
        <div style={{
          display: 'flex',
          gap: '16px',
          alignItems: 'center',
          padding: '12px',
          background: '#1e293b',
          borderRadius: '8px',
          border: '1px solid #334155',
          fontSize: '13px'
        }}>
          {/* Multimodal Blending Toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: '#cbd5e1' }}>
              <input
                type="checkbox"
                checked={useMultimodal}
                onChange={(e) => setUseMultimodal(e.target.checked)}
                disabled={!imageFile || !query.trim()}
                style={{ accentColor: '#0284c7' }}
              />
              <span>Multimodal Blend (Text + Image)</span>
            </label>
            {useMultimodal && (
              <span style={{ fontSize: '11px', color: '#94a3b8', background: '#0f172a', padding: '2px 6px', borderRadius: '4px' }}>
                50% Text + 50% Image
              </span>
            )}
          </div>

          {/* Date Range Selection */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderLeft: '1px solid #334155', paddingLeft: '16px' }}>
            <span style={{ color: '#cbd5e1' }}>Date Range:</span>
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
              style={{
                padding: '4px 8px',
                borderRadius: '4px',
                border: '1px solid #475569',
                background: '#0f172a',
                color: '#f8fafc',
                fontSize: '12px'
              }}
            />
            <span style={{ color: '#64748b' }}>→</span>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
              style={{
                padding: '4px 8px',
                borderRadius: '4px',
                border: '1px solid #475569',
                background: '#0f172a',
                color: '#f8fafc',
                fontSize: '12px'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
