import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import PromptsPage from './pages/PromptsPage'
import ConfigPage from './pages/ConfigPage'
import LogsPage from './pages/LogsPage'
import { Toaster } from 'sonner'

function App() {
  return (
    <div className="min-h-screen bg-background">
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/prompts" element={<PromptsPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/logs" element={<LogsPage />} />
        </Routes>
      </Layout>
      <Toaster 
        position="top-right" 
        richColors 
        closeButton
        duration={4000}
      />
    </div>
  )
}

export default App