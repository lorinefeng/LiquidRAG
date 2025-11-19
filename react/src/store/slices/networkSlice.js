import { createSlice } from '@reduxjs/toolkit'

const slice = createSlice({
  name: 'network',
  initialState: { connected: true },
  reducers: {
    setConnected(state, action) { state.connected = !!action.payload }
  }
})

export const { setConnected } = slice.actions
export default slice.reducer