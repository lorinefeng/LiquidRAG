import axios from 'axios'
import store from '../store'
import { setConnected } from '../store/slices/networkSlice'

const api = axios.create({ baseURL: '' })

api.interceptors.response.use(
  (resp) => { store.dispatch(setConnected(true)); return resp },
  (error) => { store.dispatch(setConnected(false)); return Promise.reject(error) }
)

export async function ping() {
  try { await api.get('/api/v1/health'); store.dispatch(setConnected(true)) }
  catch { store.dispatch(setConnected(false)) }
}

export default api