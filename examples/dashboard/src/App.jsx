import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import PwsaPage from './pages/PwsaPage'
import FinancePage from './pages/FinancePage'
import LlmPage from './pages/LlmPage'
import SettingsPage from './pages/SettingsPage'
import { ApiProvider } from './contexts/ApiContext'

function App() {
  return (
    <ApiProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="pwsa" element={<PwsaPage />} />
            <Route path="finance" element={<FinancePage />} />
            <Route path="llm" element={<LlmPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ApiProvider>
  )
}

export default App
