import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // onnxruntime-web trae su propio .wasm: no lo pre-empaquetamos
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
