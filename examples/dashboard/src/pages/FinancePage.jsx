import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useApi } from '../contexts/ApiContext'
import { TrendingUp, AlertTriangle } from 'lucide-react'

const FinancePage = () => {
  const { api } = useApi()
  const [result, setResult] = useState(null)

  const optimizeMutation = useMutation({
    mutationFn: (data) => api.optimizePortfolio(data),
    onSuccess: (response) => setResult(response.data),
  })

  const handleOptimize = () => {
    optimizeMutation.mutate({
      assets: [
        { symbol: 'AAPL', expected_return: 0.12, volatility: 0.25, current_price: 150.0 },
        { symbol: 'GOOGL', expected_return: 0.15, volatility: 0.30, current_price: 2800.0 },
        { symbol: 'MSFT', expected_return: 0.13, volatility: 0.22, current_price: 380.0 },
      ],
      constraints: {
        max_position_size: 0.5,
        min_position_size: 0.1,
        max_total_risk: 0.20,
      },
      objective: 'maximize_sharpe',
    })
  }

  return (
    <div className="px-4 py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <TrendingUp className="text-primary-400" />
          Finance - Portfolio Optimization
        </h1>
        <p className="text-slate-400 mt-1">Optimize asset allocation and risk management</p>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Optimize Portfolio</h3>
        <div className="space-y-4">
          <div className="p-3 bg-slate-700 rounded-lg text-sm">
            <p className="text-slate-300 mb-2">Portfolio Assets:</p>
            <ul className="space-y-1 text-slate-400">
              <li>• AAPL: 12% expected return, 25% volatility</li>
              <li>• GOOGL: 15% expected return, 30% volatility</li>
              <li>• MSFT: 13% expected return, 22% volatility</li>
            </ul>
          </div>

          <button
            onClick={handleOptimize}
            disabled={optimizeMutation.isPending}
            className="btn-primary w-full"
          >
            {optimizeMutation.isPending ? 'Optimizing...' : 'Optimize Portfolio'}
          </button>

          {optimizeMutation.isError && (
            <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg flex items-start gap-3">
              <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="text-red-400 font-medium">Error</p>
                <p className="text-red-300 text-sm mt-1">
                  {optimizeMutation.error?.message || 'Failed to optimize portfolio'}
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="p-4 bg-slate-700 rounded-lg">
              <h4 className="text-white font-medium mb-3">Optimization Result</h4>
              <div className="space-y-3">
                <div>
                  <p className="text-slate-400 text-sm mb-2">Optimal Weights:</p>
                  {result.data?.weights?.map((w, idx) => (
                    <div key={idx} className="flex justify-between text-sm mb-1">
                      <span className="text-slate-300">{w.symbol}:</span>
                      <span className="text-white font-mono">{(w.weight * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
                <div className="pt-3 border-t border-slate-600 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Expected Return:</span>
                    <span className="text-green-400 font-medium">
                      {result.data?.expected_return ? `${(result.data.expected_return * 100).toFixed(2)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Expected Risk:</span>
                    <span className="text-yellow-400 font-medium">
                      {result.data?.expected_risk ? `${(result.data.expected_risk * 100).toFixed(2)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sharpe Ratio:</span>
                    <span className="text-blue-400 font-medium">
                      {result.data?.sharpe_ratio?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default FinancePage
