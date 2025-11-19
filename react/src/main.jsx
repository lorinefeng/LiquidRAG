import React from 'react'
import { createRoot } from 'react-dom/client'
import { ConfigProvider, theme } from 'antd'
import { App as AntdApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { Provider } from 'react-redux'
import store from './store'
import App from './App'
import 'antd/dist/reset.css'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <Provider store={store}>
    <ConfigProvider locale={zhCN} theme={{ algorithm: theme.darkAlgorithm }}>
      <AntdApp>
        <App />
      </AntdApp>
    </ConfigProvider>
  </Provider>
)