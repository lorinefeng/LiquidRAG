function qs(sel) { return document.querySelector(sel) }
const inputEl = qs('#poda .input')
const sendBtn = qs('#sendBtn')
const statusBar = qs('#statusBar')
const loader = qs('#loaderContainer')
const messageList = qs('#messageList')
const sourceList = qs('#sourceList')

function showLoader(show) { loader.classList.toggle('hidden', !show) }

function addMessage(text, role) {
  const div = document.createElement('div')
  div.className = `message ${role}`
  div.textContent = text
  messageList.appendChild(div)
  messageList.scrollTop = messageList.scrollHeight
}

function renderSources(sources) {
  sourceList.innerHTML = ''
  if (!Array.isArray(sources) || sources.length === 0) return
  sources.forEach(s => {
    const item = document.createElement('div')
    item.className = 'source-item'
    const title = `${s.source || ''}  相似度: ${typeof s.similarity === 'number' ? s.similarity.toFixed(3) : s.similarity || ''}`
    const p1 = document.createElement('div')
    p1.textContent = title
    const p2 = document.createElement('div')
    p2.textContent = s.chunk_text || ''
    item.appendChild(p1)
    item.appendChild(p2)
    sourceList.appendChild(item)
  })
}

async function sendQuery() {
  const q = (inputEl.value || '').trim()
  if (!q) return
  addMessage(q, 'user')
  inputEl.value = ''
  showLoader(true)
  try {
    const r = await window.api.ask({ query: q, top_k: 5, return_sources: true })
    addMessage(r.answer || '没有返回答案', 'assistant')
    renderSources(r.sources || [])
  } catch (e) {
    addMessage('请求失败，请检查后端是否已启动。', 'assistant')
  } finally {
    showLoader(false)
  }
}

async function updateStatus() {
  try {
    const s = await window.api.status()
    const gm = s.gpu_memory || {}
    const used = typeof gm.used_gb === 'number' ? gm.used_gb.toFixed(2) : gm.used_gb
    const total = typeof gm.total_gb === 'number' ? gm.total_gb.toFixed(2) : gm.total_gb
    const ready = s.pipeline_initialized ? '已就绪' : '加载中'
    statusBar.textContent = `状态: ${ready} | 模型: ${s.model || '-'} | 嵌入: ${s.embedding_model || '-'} | 显存: ${used || '-'} / ${total || '-' } GB`
  } catch (_) {
    statusBar.textContent = '后端未连接'
  }
}

document.addEventListener('DOMContentLoaded', () => {
  sendBtn.addEventListener('click', sendQuery)
  inputEl.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') sendQuery() })
  updateStatus()
  setInterval(updateStatus, 3000)
})