import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from '../../utils/api'

export const ask = createAsyncThunk('chat/ask', async ({ text }, { rejectWithValue }) => {
  try {
    const res = await api.post('/api/v1/ask', { query: text, return_sources: true })
    return res.data
  } catch (e) { return rejectWithValue(e?.response?.data || { message: '请求失败' }) }
})

const slice = createSlice({
  name: 'chat',
  initialState: { messages: [], generating: false },
  reducers: {
    addUser(state, action) { state.messages.push({ role: 'user', content: action.payload, time: Date.now() }) },
    clear(state) { state.messages = [] }
  },
  extraReducers: (b) => {
    b.addCase(ask.pending, (state) => { state.generating = true })
     .addCase(ask.fulfilled, (state, action) => {
        state.generating = false
        state.messages.push({ role: 'assistant', content: action.payload.answer, sources: action.payload.sources, time: Date.now() })
      })
     .addCase(ask.rejected, (state, action) => { state.generating = false })
  }
})

export const { addUser, clear } = slice.actions
export default slice.reducer