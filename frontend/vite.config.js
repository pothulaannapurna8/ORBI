import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/search': 'http://localhost:8000',
      '/review': 'http://localhost:8000',
      '/results': 'http://localhost:8000',
      '/tiles': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/static': 'http://localhost:8000'
    }
  }
});
