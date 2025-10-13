import React, { useState } from 'react'
import { useApi } from '../contexts/ApiContext'
import { Settings, Save, Key, Globe } from 'lucide-react'

const SettingsPage = () => {
  const { apiKey, setApiKey, baseUrl, setBaseUrl } = useApi()
  const [localApiKey, setLocalApiKey] = useState(apiKey)
  const [localBaseUrl, setLocalBaseUrl] = useState(baseUrl)
  const [saved, setSaved] = useState(false)

  const handleSave = (e) => {
    e.preventDefault()
    setApiKey(localApiKey)
    setBaseUrl(localBaseUrl)
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  return (
    <div className="px-4 py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <Settings className="text-primary-400" />
          Settings
        </h1>
        <p className="text-slate-400 mt-1">Configure your API connection</p>
      </div>

      <div className="card max-w-2xl">
        <h3 className="text-lg font-semibold text-white mb-6">API Configuration</h3>
        <form onSubmit={handleSave} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
              <Globe size={16} />
              API Base URL
            </label>
            <input
              type="url"
              value={localBaseUrl}
              onChange={(e) => setLocalBaseUrl(e.target.value)}
              className="input"
              placeholder="http://localhost:8080"
              required
            />
            <p className="text-xs text-slate-400 mt-2">
              The base URL of your PRISM-AI API server
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
              <Key size={16} />
              API Key
            </label>
            <input
              type="password"
              value={localApiKey}
              onChange={(e) => setLocalApiKey(e.target.value)}
              className="input font-mono"
              placeholder="Enter your API key"
              required
            />
            <p className="text-xs text-slate-400 mt-2">
              Your authentication key for accessing the API
            </p>
          </div>

          <div className="flex items-center gap-4 pt-4 border-t border-slate-700">
            <button type="submit" className="btn-primary flex items-center gap-2">
              <Save size={18} />
              Save Configuration
            </button>
            {saved && (
              <span className="text-green-400 text-sm font-medium">
                ✓ Settings saved successfully
              </span>
            )}
          </div>
        </form>

        <div className="mt-8 pt-8 border-t border-slate-700">
          <h4 className="text-white font-medium mb-4">Quick Test</h4>
          <p className="text-slate-300 text-sm mb-4">
            After saving your configuration, visit the Dashboard to verify your connection.
          </p>
          <a href="/" className="btn-secondary text-sm">
            Go to Dashboard
          </a>
        </div>
      </div>

      <div className="card max-w-2xl bg-blue-900/10 border-blue-700">
        <h4 className="text-blue-300 font-medium mb-2">Getting Started</h4>
        <ul className="text-sm text-slate-300 space-y-2">
          <li>1. Start your PRISM-AI API server on <code className="text-blue-400">localhost:8080</code></li>
          <li>2. Enter your API key above</li>
          <li>3. Click "Save Configuration"</li>
          <li>4. Navigate to the Dashboard to see real-time metrics</li>
        </ul>
      </div>
    </div>
  )
}

export default SettingsPage
