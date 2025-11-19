import { configureStore } from '@reduxjs/toolkit'
import chat from './slices/chatSlice'
import network from './slices/networkSlice'
import kb from './slices/kbSlice'
import upload from './slices/uploadSlice'

export default configureStore({
  reducer: { chat, network, kb, upload }
})