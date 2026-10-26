import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': { target: process.env.VITE_API_TARGET || 'http://localhost:4000', changeOrigin: false } } },
  test: { environment: 'jsdom', globals: true, include: ['src/**/*.test.jsx'] },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 700 },
});
