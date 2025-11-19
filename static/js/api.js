const API_BASE = 'http://127.0.0.1:8000/api/v1'

async function ask({ query, top_k = 5, return_sources = true }) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k, return_sources })
  })
  if (!res.ok) throw new Error('ASK_FAILED')
  return res.json()
}

async function status() {
  const res = await fetch(`${API_BASE}/status`)
  if (!res.ok) throw new Error('STATUS_FAILED')
  return res.json()
}

async function clearHistory() {
  const res = await fetch(`${API_BASE}/clear-history`, { method: 'POST' })
  if (!res.ok) throw new Error('CLEAR_HISTORY_FAILED')
  return res.json()
}

async function upload(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('UPLOAD_FAILED')
  return res.json()
}

async function documents() {
  const res = await fetch(`${API_BASE}/documents`)
  if (!res.ok) throw new Error('DOCUMENTS_FAILED')
  return res.json()
}

async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('DELETE_DOC_FAILED')
  return res.json()
}

window.api = { ask, status, clearHistory, upload, documents, deleteDocument }