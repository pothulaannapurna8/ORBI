import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar.jsx';
import FilterPanel from './components/FilterPanel.jsx';
import MapView from './components/MapView.jsx';
import ResultCard from './components/ResultCard.jsx';
import BeforeAfterViewer from './components/BeforeAfterViewer.jsx';
import ReviewPanel from './components/ReviewPanel.jsx';
import EvidenceExport from './components/EvidenceExport.jsx';
import { apiFetch } from './api.js';

export default function App() {
  const [query, setQuery] = useState("Find newly built structures near rivers between 2024 and 2026");
  const [imageFile, setImageFile] = useState(null);
  const [aoi, setAoi] = useState('ALL');
  const [sensor, setSensor] = useState('Sentinel-2 L2A');
  const [topK, setTopK] = useState(20);
  const [showSuppressed, setShowSuppressed] = useState(false);
  const [useMultimodal, setUseMultimodal] = useState(false);
  const [dateRange, setDateRange] = useState({ start: '2024-01-01', end: '2026-12-31' });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [similarResults, setSimilarResults] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [error, setError] = useState('');

  // Perform initial search on mount
  useEffect(() => {
    handleSearch();
  }, []);

  const handleSearch = async () => {
    setLoading(true);
    setError('');
    try {
      let res;
      if (useMultimodal && imageFile && query.trim()) {
        // Multimodal search with blending
        const formData = new FormData();
        formData.append('query', query);
        formData.append('file', imageFile);
        formData.append('top_k', topK.toString());
        formData.append('date_start', dateRange.start);
        formData.append('date_end', dateRange.end);
          formData.append('sensor', sensor !== 'ALL' ? sensor : '');
        res = await apiFetch('/search/multimodal', {
          method: 'POST',
          body: formData
        });
      } else if (imageFile) {
        // Pure image search
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('top_k', topK.toString());
        formData.append('date_start', dateRange.start);
        formData.append('date_end', dateRange.end);
          formData.append('sensor', sensor !== 'ALL' ? sensor : '');
        res = await apiFetch('/search/image', {
          method: 'POST',
          body: formData
        });
      } else {
        // Natural language text search
        res = await apiFetch('/search/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query,
            top_k: topK,
            sensor: sensor !== 'ALL' ? sensor : null,
            date_start: dateRange.start,
            date_end: dateRange.end
          })
        });
      }

      if (res.ok) {
        const data = await res.json();
        const rawResults = data.results || [];
        setResults(rawResults);
        setSimilarResults([]);
        if (rawResults.length > 0) {
          setSelectedResult(rawResults[0]);
        } else {
          setSelectedResult(null);
        }
      } else {
        setError(`Search request failed (${res.status}). Check that the backend is running.`);
      }
    } catch (err) {
      console.error('Search error:', err);
      setError('Backend unavailable. Start FastAPI on port 8000 and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFindSimilar = async (result) => {
    setSimilarLoading(true);
    try {
      const response = await apiFetch(`/results/${result.tile_id}/similar?top_k=10`);
      if (!response.ok) throw new Error(`Similar-site request failed: ${response.status}`);
      const data = await response.json();
      setSimilarResults((data.results || []).map(item => ({ ...item, isSimilar: true })));
    } catch (error) {
      console.error('Similar-site search error:', error);
      setSimilarResults([]);
    } finally {
      setSimilarLoading(false);
    }
  };

  // Filter results for suppressed toggle
  const filteredResults = results.filter(r => {
    if (!showSuppressed && r.change_confidence < 0.40) return false;
    if (aoi === 'URBAN' && !r.location_key.includes('p12_950')) return false;
    if (aoi === 'RIVER' && !r.location_key.includes('p13_')) return false;
    return true;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', background: '#0f172a' }}>
      {/* 1. Header & Search Bar */}
      <header style={{
        background: '#1e293b',
        borderBottom: '1px solid #334155',
        padding: '12px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '24px',
        zIndex: 20
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '280px' }}>
          <span style={{ fontSize: '20px' }}>🛰️</span>
          <div>
            <h1 style={{ fontSize: '15px', fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.02em' }}>
              PS26227 EO Intelligence
            </h1>
            <p style={{ fontSize: '11px', color: '#94a3b8' }}>
              Semantic Retrieval & Multi-Temporal Change
            </p>
          </div>
        </div>

        <SearchBar
          query={query}
          setQuery={setQuery}
          imageFile={imageFile}
          setImageFile={setImageFile}
          onSearch={handleSearch}
          loading={loading}
          useMultimodal={useMultimodal}
          setUseMultimodal={setUseMultimodal}
          dateRange={dateRange}
          setDateRange={setDateRange}
        />
      </header>

      {/* 2. Main Content Grid */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left: Filters & Ranked Candidate List */}
        <div style={{ display: 'flex', width: '480px', borderRight: '1px solid #334155' }}>
          <FilterPanel
            aoi={aoi}
            setAoi={setAoi}
            sensor={sensor}
            setSensor={setSensor}
            topK={topK}
            setTopK={setTopK}
            showSuppressed={showSuppressed}
            setShowSuppressed={setShowSuppressed}
          />

          <div style={{
            flex: 1,
            background: '#0f172a',
            display: 'flex',
            flexDirection: 'column',
            overflowY: 'auto',
            padding: '12px',
            gap: '10px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '4px' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' }}>
                Ranked Candidates ({filteredResults.length})
              </span>
            </div>

            {error ? (
              <div style={{ color: '#fbbf24', textAlign: 'center', padding: '32px 12px', fontSize: '13px' }}>{error}</div>
            ) : loading ? (
              [1, 2, 3].map(item => <div key={item} className="result-skeleton" />)
            ) : filteredResults.length === 0 ? (
              <div style={{ color: '#64748b', textAlign: 'center', padding: '40px 10px', fontSize: '13px' }}>
                No results above the confidence threshold.
                {results.length > filteredResults.length && ` ${results.length - filteredResults.length} candidate(s) were suppressed as likely false alarms.`}
                {results.length > filteredResults.length && (
                  <button type="button" onClick={() => setShowSuppressed(true)} style={{ display: 'block', margin: '12px auto 0', background: 'transparent', border: '1px solid #475569', borderRadius: '4px', color: '#67e8f9', padding: '6px 10px', cursor: 'pointer' }}>
                    Show suppressed
                  </button>
                )}
              </div>
            ) : (
              filteredResults.map(item => (
                <ResultCard
                  key={item.tile_id || item.location_key}
                  result={item}
                  isSelected={selectedResult && selectedResult.location_key === item.location_key}
                  onSelect={() => setSelectedResult(item)}
                  onFindSimilar={handleFindSimilar}
                />
              ))
            )}
            {similarResults.length > 0 && (
              <div style={{ borderTop: '1px solid #334155', paddingTop: '12px', marginTop: '4px' }}>
                <div style={{ color: '#67e8f9', fontSize: '12px', fontWeight: 700, marginBottom: '8px', textTransform: 'uppercase' }}>
                  Similar to selected site ({similarResults.length})
                </div>
                {similarResults.map(item => (
                  <ResultCard
                    key={`similar-${item.tile_id}`}
                    result={item}
                    isSelected={false}
                    isSimilar
                    onSelect={() => setSelectedResult(item)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center: Interactive MapLibre GL Map */}
        <div style={{ flex: 1, position: 'relative', height: '100%' }}>
          <MapView
            results={filteredResults}
            selectedResult={selectedResult}
            onSelectResult={setSelectedResult}
          />
        </div>

        {/* Right: Detailed Inspection & Evidence Panel */}
        {selectedResult && (
          <div style={{
            width: '440px',
            background: '#1e293b',
            borderLeft: '1px solid #334155',
            display: 'flex',
            flexDirection: 'column',
            padding: '18px',
            overflowY: 'auto',
            gap: '16px'
          }}>
            {/* Header / Badges */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc' }}>
                  {selectedResult.location_key}
                </h3>
                <span className={selectedResult.change_confidence >= 0.70 ? 'badge-high' : (selectedResult.change_confidence >= 0.40 ? 'badge-review' : 'badge-suppressed')}
                      style={{ fontSize: '11px', fontWeight: 700, padding: '3px 8px', borderRadius: '4px', textTransform: 'uppercase' }}>
                  {selectedResult.change_type}
                </span>
              </div>
              <p style={{ fontSize: '12px', color: '#94a3b8' }}>
                Coordinates: [{selectedResult.coordinates?.[1]?.toFixed(4)}°N, {selectedResult.coordinates?.[0]?.toFixed(4)}°E]
              </p>
            </div>

            {/* Score Metrics Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '10px',
              background: '#0f172a',
              padding: '12px',
              borderRadius: '8px',
              border: '1px solid #334155'
            }}>
              <div>
                <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>Change Confidence Score</span>
                <span style={{ fontSize: '18px', fontWeight: 700, color: selectedResult.change_confidence >= 0.70 ? '#4ade80' : (selectedResult.change_confidence >= 0.40 ? '#fbbf24' : '#f87171') }}>
                  {selectedResult.change_confidence}
                </span>
              </div>
              <div>
                <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>Semantic Similarity</span>
                <span style={{ fontSize: '18px', fontWeight: 700, color: '#38bdf8' }}>
                  {selectedResult.similarity}
                </span>
              </div>
              <div style={{ gridColumn: 'span 2', borderTop: '1px solid #1e293b', paddingTop: '8px', marginTop: '4px' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8', display: 'block' }}>Estimated Earliest Change Date:</span>
                <span style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc' }}>
                  📅 {selectedResult.earliest_change_date || 'N/A'}
                </span>
              </div>
            </div>

            {/* Before/After Viewer */}
            <BeforeAfterViewer result={selectedResult} />

            {/* Model Provenance Card */}
            <div style={{
              background: '#0f172a',
              borderRadius: '8px',
              border: '1px solid #334155',
              padding: '12px',
              fontSize: '12px',
              color: '#cbd5e1'
            }}>
              <div style={{ fontWeight: 600, color: '#94a3b8', marginBottom: '6px', textTransform: 'uppercase', fontSize: '11px' }}>
                Audit & Model Provenance
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span>Change Detector:</span>
                <span style={{ color: '#f8fafc', fontWeight: 500 }}>{selectedResult.provenance?.model_version || 'Open-CD SNUNet'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span>AROSICS Co-Reg Shift:</span>
                <span style={{ color: '#f8fafc', fontWeight: 500 }}>{selectedResult.provenance?.registration_shift_px} px</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>s2cloudless Clear Quality:</span>
                <span style={{ color: '#f8fafc', fontWeight: 500 }}>{selectedResult.provenance?.cloud_mask_quality}</span>
              </div>
            </div>

            {/* Analyst Review Form */}
            <ReviewPanel result={selectedResult} onFindSimilar={handleFindSimilar} similarLoading={similarLoading} />

            {/* Evidence Package Exporter */}
            <EvidenceExport result={selectedResult} />
          </div>
        )}
      </div>
    </div>
  );
}
