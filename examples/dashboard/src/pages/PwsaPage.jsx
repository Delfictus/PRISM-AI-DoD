import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useApi } from '../contexts/ApiContext'
import { Shield, AlertTriangle } from 'lucide-react'

const PwsaPage = () => {
  const { api } = useApi()
  const [svId, setSvId] = useState('42')
  const [result, setResult] = useState(null)

  const detectMutation = useMutation({
    mutationFn: (data) => api.detectThreat(data),
    onSuccess: (response) => setResult(response.data),
  })

  const handleDetect = () => {
    detectMutation.mutate({
      sv_id: parseInt(svId),
      timestamp: Math.floor(Date.now() / 1000),
      ir_frame: {
        width: 640,
        height: 480,
        centroid_x: 320.0,
        centroid_y: 240.0,
        hotspot_count: 5,
      },
    })
  }

  return (
    <div className="px-4 py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <Shield className="text-primary-400" />
          PWSA - Threat Detection
        </h1>
        <p className="text-slate-400 mt-1">Passive Wide-area Surveillance & Assessment</p>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Detect Threat</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Space Vehicle ID
            </label>
            <input
              type="number"
              value={svId}
              onChange={(e) => setSvId(e.target.value)}
              className="input"
              placeholder="42"
            />
          </div>

          <button
            onClick={handleDetect}
            disabled={detectMutation.isPending}
            className="btn-primary w-full"
          >
            {detectMutation.isPending ? 'Detecting...' : 'Detect Threat'}
          </button>

          {detectMutation.isError && (
            <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg flex items-start gap-3">
              <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="text-red-400 font-medium">Error</p>
                <p className="text-red-300 text-sm mt-1">
                  {detectMutation.error?.message || 'Failed to detect threat'}
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="p-4 bg-slate-700 rounded-lg">
              <h4 className="text-white font-medium mb-3">Detection Result</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Threat ID:</span>
                  <span className="text-white font-mono">{result.data?.threat_id || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Type:</span>
                  <span className="text-white">{result.data?.threat_type || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Confidence:</span>
                  <span className="text-white">
                    {result.data?.confidence ? `${(result.data.confidence * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PwsaPage
