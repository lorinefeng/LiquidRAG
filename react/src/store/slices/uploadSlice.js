import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import api from '../../utils/api'

export const uploadFile = createAsyncThunk('upload/file', async ({ file }, { rejectWithValue }) => {
  try {
    const form = new FormData()
    form.append('file', file)
    const res = await api.post('/api/v1/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {}
    })
    return res.data
  } catch (e) { return rejectWithValue(e?.response?.data || { message: '上传失败' }) }
})

const slice = createSlice({
  name: 'upload',
  initialState: { progress: 0, lastResult: null },
  reducers: {
    setProgress(state, action) { state.progress = action.payload }
  },
  extraReducers: (b) => {
    b.addCase(uploadFile.pending, (state) => { state.progress = 0 })
     .addCase(uploadFile.fulfilled, (state, action) => { state.lastResult = action.payload; state.progress = 100 })
     .addCase(uploadFile.rejected, (state) => { state.progress = 0 })
  }
})

export const { setProgress } = slice.actions
export default slice.reducer