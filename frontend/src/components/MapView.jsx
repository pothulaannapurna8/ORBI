import React, { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';

export default function MapView({
  results,
  selectedResult,
  onSelectResult
}) {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const markersRef = useRef([]);

  useEffect(() => {
    if (map.current) return;

    // Initialize MapLibre GL map
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap Contributors'
          }
        },
        layers: [
          {
            id: 'osm-layer',
            type: 'raster',
            source: 'osm',
            minzoom: 0,
            maxzoom: 19
          }
        ]
      },
      center: [77.5946, 12.9716], // Bangalore center
      zoom: 10
    });

    map.current.addControl(new maplibregl.NavigationControl(), 'top-right');
  }, []);

  // Update Markers on results change
  useEffect(() => {
    if (!map.current) return;

    // Clear old markers
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    if (!results || results.length === 0) return;

    // Bounds to fit
    const bounds = new maplibregl.LngLatBounds();

    results.forEach(item => {
      const [lon, lat] = item.coordinates || [77.5946, 12.9716];
      bounds.extend([lon, lat]);

      // Color coding by Change Confidence Score
      let color = '#22c55e'; // High (>= 0.70)
      if (item.change_confidence < 0.40) {
        color = '#ef4444'; // Suppressed (< 0.40)
      } else if (item.change_confidence < 0.70) {
        color = '#f59e0b'; // Review (0.40 - 0.70)
      }

      const isSelected = selectedResult && selectedResult.location_key === item.location_key;

      const el = document.createElement('div');
      el.style.width = isSelected ? '26px' : '18px';
      el.style.height = isSelected ? '26px' : '18px';
      el.style.backgroundColor = color;
      el.style.borderRadius = '50%';
      el.style.border = isSelected ? '3px solid white' : '2px solid #0f172a';
      el.style.boxShadow = '0 2px 8px rgba(0,0,0,0.6)';
      el.style.cursor = 'pointer';
      el.style.transition = 'all 0.2s ease';

      el.addEventListener('click', () => {
        onSelectResult(item);
      });

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([lon, lat])
        .addTo(map.current);

      markersRef.current.push(marker);
    });

    if (!bounds.isEmpty()) {
      map.current.fitBounds(bounds, { padding: 80, maxZoom: 14 });
    }
  }, [results, selectedResult]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />

      {/* Map legend overlay */}
      <div style={{
        position: 'absolute',
        bottom: '24px',
        left: '24px',
        background: 'rgba(15, 23, 42, 0.9)',
        backdropFilter: 'blur(8px)',
        border: '1px solid #334155',
        borderRadius: '8px',
        padding: '10px 14px',
        fontSize: '12px',
        color: '#cbd5e1',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        zIndex: 10
      }}>
        <div style={{ fontWeight: 600, color: '#f8fafc', marginBottom: '2px' }}>Change Confidence Score</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#22c55e' }} />
          <span>High Confidence (&ge; 0.70)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#f59e0b' }} />
          <span>Review Required (0.40 - 0.70)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444' }} />
          <span>Suppressed / False Alarm (&lt; 0.40)</span>
        </div>
      </div>
    </div>
  );
}
