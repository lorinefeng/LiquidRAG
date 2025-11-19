import React from 'react'
import { Alert } from 'antd'

export default class ErrorBoundary extends React.Component {
  constructor(props){
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error){ return { hasError: true, error } }
  componentDidCatch(error, info){ }
  render(){
    if (this.state.hasError) {
      return <Alert type="error" showIcon message="页面发生错误" description={String(this.state.error||'未知错误')} />
    }
    return this.props.children
  }
}