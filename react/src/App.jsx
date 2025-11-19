import React, { useEffect, useState } from 'react'
import { Layout, Input, Button, Tag, Tabs, Upload, Progress, Table, Space, Modal, Alert, App as AntdApp } from 'antd'
import { useDispatch, useSelector } from 'react-redux'
import { addUser, ask, clear } from './store/slices/chatSlice'
import { uploadFile } from './store/slices/uploadSlice'
import { fetchDocs, deleteDoc, batchDelete, setSelected, rebuildIndex } from './store/slices/kbSlice'
import api, { ping } from './utils/api'
import ErrorBoundary from './components/ErrorBoundary'
import logo from './assets/logo.svg'

const { Header, Content } = Layout

function ChatPage() {
  const dispatch = useDispatch()
  const { messages, generating } = useSelector(s => s.chat)
  const { connected } = useSelector(s => s.network)
  const [text, setText] = useState('')

  const send = () => {
    if (!text.trim()) return
    dispatch(addUser(text))
    dispatch(ask({ text }))
    setText('')
  }
  useEffect(()=>{
    const id = setInterval(()=>{ ping() }, 3000)
    ping()
    return ()=>clearInterval(id)
  },[])

  return (
    <>
      <div className="status-bar">状态：{connected ? '已连接' : '断开'} | 模式：local-LLM</div>
      <div className="chat-container">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              <div>{m.content}</div>
              <span className="ts">{m.time ? new Date(m.time).toLocaleTimeString() : new Date().toLocaleTimeString()}</span>
              {m.sources && (
                <div style={{ marginTop: 8 }}>
                  {(m.sources||[]).slice(0,3).map((s, idx) => (
                    <Tag key={idx}>{s.source} | {s.similarity?.toFixed(3)}</Tag>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="bottom-input">
        <div className="bottom-inner">
          <div className="input-row">
            <InputBox value={text} onChange={setText} onSubmit={send} />
          </div>
          <div className="actions">
            <Button type="primary" onClick={send}>发送</Button>
            <Button onClick={()=>dispatch(clear())}>清空</Button>
          </div>
        </div>
      </div>
      <div className="generating"><GeneratingIndicator visible={generating} /></div>
    </>
  )
}

function KbPage() {
  const dispatch = useDispatch()
  const { items, page, page_size, total, selected } = useSelector(s => s.kb)
  const { progress } = useSelector(s => s.upload)
  const [q, setQ] = useState('')
  const [preview, setPreview] = useState(null)
  const [lastError, setLastError] = useState(null)
  const [failedFile, setFailedFile] = useState(null)
  const { message } = AntdApp.useApp()

  useEffect(()=>{ dispatch(fetchDocs({ page:1, page_size:20, q:'' })).unwrap().catch((e)=>message.error(e?.detail||'获取文档失败')) }, [dispatch])

  const props = {
    beforeUpload: (file) => {
      const ok = ['.md','.txt'].includes((file.name||'').toLowerCase().slice((file.name||'').lastIndexOf('.')))
      if (!ok) { message.error('仅支持 .md / .txt'); return Upload.LIST_IGNORE }
      dispatch(uploadFile({ file })).unwrap()
        .then(()=>{ message.success('上传入库成功'); setLastError(null); setFailedFile(null); dispatch(fetchDocs({ page, page_size, q })) })
        .catch((e)=>{ const msg = e?.detail || e?.message || '上传失败'; setLastError(msg); setFailedFile(file); message.error(msg) })
      return Upload.LIST_IGNORE
    }
  }

  const retryUpload = () => {
    if (!failedFile) return
    dispatch(uploadFile({ file: failedFile })).unwrap()
      .then(()=>{ message.success('重试成功'); setLastError(null); setFailedFile(null); dispatch(fetchDocs({ page, page_size, q })) })
      .catch((e)=>{ const msg = e?.detail || e?.message || '重试失败'; setLastError(msg); message.error(msg) })
  }

  const columns = [
    { title: '源文件', dataIndex: 'source' },
    { title: '类型', dataIndex: 'file_type', width: 120 },
    { title: '块数', dataIndex: 'chunks', width: 100 },
    { title: '操作', key: 'op', width: 220, render: (_, r) => (
      <Space>
        <Button onClick={()=>setPreview(r.preview)}>预览</Button>
        <Button danger onClick={()=>{
          Modal.confirm({
            title: '确认要删除此文件吗？',
            okText: '确认',
            cancelText: '取消',
            okButtonProps: { danger: true },
            onOk: () => {
              return dispatch(deleteDoc({ source: r.source })).unwrap()
                .then(()=>{ message.success('删除成功'); dispatch(fetchDocs({ page, page_size, q })) })
                .catch((e)=>{ message.error(e?.detail || e?.message || '删除失败') })
            }
          })
        }}>删除</Button>
      </Space>
    ) }
  ]

  return (
    <div className="kb-container">
      <Space className="kb-toolbar" style={{ marginBottom: 12 }}>
        <Input placeholder="搜索源文件" value={q} onChange={(e)=>setQ(e.target.value)} style={{ width: 240 }} />
        <Button onClick={()=>dispatch(fetchDocs({ page:1, page_size, q })).unwrap().catch((e)=>message.error(e?.detail||'获取文档失败'))}>搜索</Button>
        <Upload {...props} showUploadList={false}><Button type="primary">上传文档</Button></Upload>
        <Button danger disabled={!selected.length} onClick={()=>{
          Modal.confirm({
            title: '确认要删除选中的文件吗？',
            okText: '确认',
            cancelText: '取消',
            okButtonProps: { danger: true },
            onOk: () => {
              return dispatch(batchDelete({ sources: selected })).unwrap()
                .then(()=>{ message.success('批量删除成功'); dispatch(fetchDocs({ page, page_size, q })) })
                .catch((e)=>{ message.error(e?.detail || e?.message || '批量删除失败') })
            }
          })
        }}>批量删除</Button>
        <Button onClick={()=>dispatch(rebuildIndex({})).then(()=>dispatch(fetchDocs({ page:1, page_size, q:'' })))}>重建索引</Button>
      </Space>
      {lastError && (
        <Alert style={{ marginBottom: 12 }} type="error" showIcon message="上传失败" description={lastError}
               action={<Button size="small" onClick={retryUpload}>重试</Button>} closable onClose={()=>setLastError(null)} />
      )}
      {progress>0 && progress<100 && <Progress percent={progress} />}
      <Table rowKey="source" columns={columns} dataSource={items} pagination={{ current: page, pageSize: page_size, total, onChange:(p,ps)=>dispatch(fetchDocs({ page:p, page_size:ps, q })) }}
             rowSelection={{ selectedRowKeys: selected, onChange:(keys)=>dispatch(setSelected(keys)) }} />
      <Modal open={!!preview} onCancel={()=>setPreview(null)} footer={null} title="内容预览"><pre style={{ whiteSpace:'pre-wrap' }}>{preview}</pre></Modal>
    </div>
  )
}

export default function App(){
  const dispatch = useDispatch()
  const [openCustomize, setOpenCustomize] = useState(false)
  const [images, setImages] = useState([])
  const [selAi, setSelAi] = useState(null)
  const [selUser, setSelUser] = useState(null)
  const { message } = AntdApp.useApp()

  useEffect(()=>{
    fetch('/images.json').then(r=>r.json()).then(d=>setImages(d.images||[])).catch(()=>{})
    fetch('/persist/avatar').then(r=>r.json()).then(d=>{
      const ai = d.ai || null
      const user = d.user || null
      setSelAi(ai)
      setSelUser(user)
      if (!ai && !user) {
        const once = localStorage.getItem('avatar.first')
        if (!once) setOpenCustomize(true)
      }
      if (ai) document.documentElement.style.setProperty('--ai-bg-image', `url(${ai})`)
      if (user) document.documentElement.style.setProperty('--user-bg-image', `url(${user})`)
    }).catch(()=>{})
  },[])

  const saveAvatars = () => {
    const payload = { ai: selAi || '', user: selUser || '' }
    fetch('/persist/avatar', { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(payload) })
      .then(()=>{
        if (selAi) document.documentElement.style.setProperty('--ai-bg-image', `url(${selAi})`)
        else document.documentElement.style.setProperty('--ai-bg-image', 'none')
        if (selUser) document.documentElement.style.setProperty('--user-bg-image', `url(${selUser})`)
        else document.documentElement.style.setProperty('--user-bg-image', 'none')
        localStorage.setItem('avatar.first', '1')
        message.success('已保存头像选择')
        setOpenCustomize(false)
      })
      .catch(()=>message.error('保存失败'))
  }

  const onTabChange = (key) => {
    if (key === 'kb') {
      dispatch(fetchDocs({ page:1, page_size:20, q:'' }))
    }
  }
  return (
      <Layout style={{ minHeight:'100vh' }}>
        <Header style={{ display:'flex', alignItems:'center' }}>
          <div className="logo"><img src={logo} alt="项目logo" /></div>
          <div className="header-title">LiquidRAG</div>
          <Button style={{ marginLeft: 12 }} onClick={()=>setOpenCustomize(true)}>个性化</Button>
          <div className="header-avatars">
            <div className="avatar"><img src={selAi||logo} alt="AI头像" /></div>
            <div className="avatar"><img src={selUser||logo} alt="用户头像" /></div>
          </div>
        </Header>
        <Content>
          <Tabs centered destroyOnHidden onChange={onTabChange}
                items={[{ key:'chat', label:'对话', children:<ChatPage/> }, { key:'kb', label:'知识库', children:<ErrorBoundary><KbPage/></ErrorBoundary> }]} />
          <Modal open={openCustomize} onCancel={()=>setOpenCustomize(false)} onOk={saveAvatars} okText="保存" cancelText="取消" title="选择头像">
            <Tabs items={[
              { key:'ai', label:'为 AI 模型选择图像', children:(
                <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(92px, 1fr))', gap:12 }}>
                  {images.map(img => (
                    <button key={img.url} onClick={()=>setSelAi(img.url)} style={{ border: selAi===img.url? '2px solid #1677ff':'1px solid #333', borderRadius:12, padding:4, background:'#111318' }}>
                      <img src={img.url} alt={img.name} style={{ width:'100%', height:92, objectFit:'cover', borderRadius:8 }} />
                    </button>
                  ))}
                </div>
              )},
              { key:'user', label:'为用户选择图像', children:(
                <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(92px, 1fr))', gap:12 }}>
                  {images.map(img => (
                    <button key={img.url} onClick={()=>setSelUser(img.url)} style={{ border: selUser===img.url? '2px solid #cf30aa':'1px solid #333', borderRadius:12, padding:4, background:'#111318' }}>
                      <img src={img.url} alt={img.name} style={{ width:'100%', height:92, objectFit:'cover', borderRadius:8 }} />
                    </button>
                  ))}
                </div>
              )}
            ]} />
          </Modal>
        </Content>
      </Layout>
  )
}
import InputBox from './components/InputBox'
import GeneratingIndicator from './components/GeneratingIndicator'
