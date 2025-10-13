import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { useApi } from '../contexts/ApiContext'
import { Activity, AlertCircle, CheckCircle, Clock } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const Dashboard = () => {
  const { api, isConfigured } = useApi()

  const { data: healthData, isLoading, error } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await api.health()
      return response.data
    },
    enabled: isConfigured,
    refetchInterval: 5000,
  })

  if (!isConfigured) {
    return (
      <div className="px-4 py-6">
        <div className="card">
          <div className="flex items-center gap-3 text-yellow-400 mb-4">
            <AlertCircle size={24} />
            <h2 className="text-xl font-semibold">Configuration Required</h2>
          </div>
          <p className="text-slate-300 mb-4">
            Please configure your API key in the Settings page to start using the dashboard.
          </p>
          <a href="/settings" className="btn-primary">
            Go to Settings
          </a>
        </div>
      </div>
    )
  }

  const mockMetrics = [
    { time: '00:00', requests: 45, latency: 23 },
    { time: '04:00', requests: 52, latency: 25 },
    { time: '08:00', requests: 78, latency: 31 },
    { time: '12:00', requests: 125, latency: 42 },
    { time: '16:00', requests: 98, latency: 35 },
    { time: '20:00', requests: 65, latency: 28 },
  ]

  return (
    <div className="px-4 py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 mt-1">Monitor your PRISM-AI API in real-time</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* API Status */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">API Status</p>
              <p className="text-2xl font-bold text-white mt-1">
                {isLoading ? (
                  'Loading...'
                ) : error ? (
                  <span className="text-red-400">Offline</span>
                ) : (
                  <span className="text-green-400">Online</span>
                )}
              </p>
            </div>
            <div className={`p-3 rounded-full ${error ? 'bg-red-900' : 'bg-green-900'}`}>
              {error ? <AlertCircle className="text-red-400" size={24} /> : <CheckCircle className="text-green-400" size={24} />}
            </div>
          </div>
          {healthData && (
            <div className="mt-4 pt-4 border-t border-slate-700">
              <p className="text-xs text-slate-400">
                Version: <span className="text-white">{healthData.version || '0.1.0'}</span>
              </p>
            </div>
          )}
        </div>

        {/* Uptime */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Uptime</p>
              <p className="text-2xl font-bold text-white mt-1">
                {healthData?.uptime_seconds ?
                  `${Math.floor(healthData.uptime_seconds / 3600)}h` :
                  '0h'
                }
              </p>
            </div>
            <div className="p-3 rounded-full bg-blue-900">
              <Clock className="text-blue-400" size={24} />
            </div>
          </div>
        </div>

        {/* Total Requests */}
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Total Requests</p>
              <p className="text-2xl font-bold text-white mt-1">1,247</p>
            </div>
            <div className="p-3 rounded-full bg-purple-900">
              <Activity className="text-purple-400" size={24} />
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-slate-700">
            <p className="text-xs text-green-400">+12% from last hour</p>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Request Rate */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Request Rate</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={mockMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '0.5rem'
                }}
              />
              <Line
                type="monotone"
                dataKey="requests"
                stroke="#0ea5e9"
                strokeWidth={2}
                dot={{ fill: '#0ea5e9' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Latency */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Average Latency (ms)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={mockMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '0.5rem'
                }}
              />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ fill: '#10b981' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
        <div className="space-y-3">
          {[
            { type: 'PWSA', action: 'Threat Detection', time: '2 min ago', status: 'success' },
            { type: 'Finance', action: 'Portfolio Optimization', time: '5 min ago', status: 'success' },
            { type: 'LLM', action: 'Query Processing', time: '8 min ago', status: 'success' },
            { type: 'PWSA', action: 'Sensor Fusion', time: '12 min ago', status: 'success' },
          ].map((activity, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
              <div className="flex items-center gap-3">
                <span className="badge-info">{activity.type}</span>
                <span className="text-white">{activity.action}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-slate-400 text-sm">{activity.time}</span>
                <span className="badge-success">Success</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
