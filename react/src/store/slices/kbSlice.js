import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from '../../utils/api'

export const fetchDocs = createAsyncThunk('kb/fetch', async ({ page = 1, page_size = 20, q = '' } = {}, { rejectWithValue }) => {
  try { const res = await api.get('/api/v1/documents', { params: { page, page_size, q } }); return res.data } 
  catch (e) { return rejectWithValue(e?.response?.data || { message: '读取失败' }) }
})

export const deleteDoc = createAsyncThunk('kb/delete', async ({ source }, { rejectWithValue }) => {
  try { const res = await api.delete(`/api/v1/documents/${encodeURIComponent(source)}`); return { source, res: res.data } } 
  catch (e) { return rejectWithValue(e?.response?.data || { message: '删除失败' }) }
})

export const batchDelete = createAsyncThunk('kb/batchDelete', async ({ sources }, { rejectWithValue }) => {
  try { const res = await api.post('/api/v1/documents/batch_delete', { sources }); return res.data } 
  catch (e) { return rejectWithValue(e?.response?.data || { message: '批量删除失败' }) }
})

export const rebuildIndex = createAsyncThunk('kb/reindex', async ({ source_dir } = {}, { rejectWithValue }) => {
  try { const res = await api.post('/api/v1/reindex', { source_dir }); return res.data } 
  catch (e) { return rejectWithValue(e?.response?.data || { message: '重建索引失败' }) }
})

const slice = createSlice({
  name: 'kb',
  initialState: { items: [], page: 1, page_size: 20, total: 0, selected: [] },
  reducers: {
    setSelected(state, action) { state.selected = action.payload }
  },
  extraReducers: (b) => {
    b.addCase(fetchDocs.fulfilled, (state, action) => {
      state.items = action.payload.items || []; state.page = action.payload.page; state.page_size = action.payload.page_size; state.total = action.payload.total
    })
  }
})

export const { setSelected } = slice.actions
export default slice.reducer