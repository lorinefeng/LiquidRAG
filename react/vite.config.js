import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react(), {
    name: 'persist-avatar-plugin',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url) return next()
        if (req.url === '/persist/avatar' && req.method === 'GET') {
          try {
            const fs = await import('fs/promises')
            const path = await import('path')
            const p = path.join(server.config.root, 'public', 'avatar.json')
            let data = '{}'
            try { data = await fs.readFile(p, 'utf-8') } catch {}
            res.setHeader('Content-Type', 'application/json')
            res.end(data || '{}')
            return
          } catch {}
        }
        if (req.url === '/persist/avatar' && req.method === 'POST') {
          try {
            const fs = await import('fs/promises')
            const path = await import('path')
            const chunks = []
            req.on('data', (c) => chunks.push(c))
            req.on('end', async () => {
              const body = Buffer.concat(chunks).toString('utf-8')
              const p = path.join(server.config.root, 'public', 'avatar.json')
              await fs.writeFile(p, body || '{}', 'utf-8')
              res.setHeader('Content-Type', 'application/json')
              res.end('{"ok":true}')
            })
            return
          } catch {}
        }
        next()
      })
    }
  }],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  },
})