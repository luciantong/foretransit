import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  define: {
    ...(process.env.VITE_API_URL
      ? {
          __API_URL__: JSON.stringify(process.env.VITE_API_URL),
        }
      : {}),
  }
})
