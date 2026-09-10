import React, { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';

export default function MapView({ results, selectedResult, onSelectResult }) {
	const mapContainer = useRef(null);
	const map = useRef(null);
	const markersRef = useRef([]);

	useEffect(() => {
		if (map.current) return;
		map.current = new maplibregl.Map({
			container: mapContainer.current,
			style: {
				version: 8,
				sources: { osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256 } },
				layers: [{ id: 'osm-layer', type: 'raster', source: 'osm', minzoom: 0, maxzoom: 19 }]
			},
			center: [77.5946, 12.9716],
			zoom: 10
		});
		map.current.addControl(new maplibregl.NavigationControl(), 'top-right');
	}, []);

	useEffect(() => {
		if (!map.current) return;
		markersRef.current.forEach(marker => marker.remove());
		markersRef.current = [];
		if (!results?.length) return;
		const bounds = new maplibregl.LngLatBounds();
		results.forEach(item => {
			const [lon, lat] = item.coordinates || [77.5946, 12.9716];
			bounds.extend([lon, lat]);
			const marker = new maplibregl.Marker()
				.setLngLat([lon, lat])
				.getElement();
			marker.addEventListener('click', () => onSelectResult(item));
			markersRef.current.push(new maplibregl.Marker({ element: marker }).setLngLat([lon, lat]).addTo(map.current));
		});
		if (!bounds.isEmpty()) map.current.fitBounds(bounds, { padding: 80, maxZoom: 14 });
	}, [results, selectedResult, onSelectResult]);

	return <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />;
}