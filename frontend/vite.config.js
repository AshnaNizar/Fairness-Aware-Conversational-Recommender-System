import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/random_user': 'http://localhost:7860',
      '/chat':        'http://localhost:7860',
      '/cot_explain': 'http://localhost:7860',
      '/cot_debug':   'http://localhost:7860',
    }
  }
})