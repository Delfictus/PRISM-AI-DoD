import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useApi } from '../contexts/ApiContext'
import { MessageSquare, AlertTriangle, Send } from 'lucide-react'

const LlmPage = () => {
  const { api } = useApi()
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState(null)

  const queryMutation = useMutation({
    mutationFn: (data) => api.queryLlm(data),
    onSuccess: (response) => setResult(response.data),
  })

  const handleQuery = (e) => {
    e.preventDefault()
    if (!prompt.trim()) return

    queryMutation.mutate({
      prompt,
      temperature: 0.7,
      max_tokens: 500,
    })
  }

  return (
    <div className="px-4 py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
          <MessageSquare className="text-primary-400" />
          LLM - Language Model
        </h1>
        <p className="text-slate-400 mt-1">Query language models and get AI-powered responses</p>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Query LLM</h3>
        <form onSubmit={handleQuery} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="input min-h-[120px] resize-none"
              placeholder="Enter your question or prompt..."
            />
          </div>

          <button
            type="submit"
            disabled={queryMutation.isPending || !prompt.trim()}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            <Send size={18} />
            {queryMutation.isPending ? 'Processing...' : 'Send Query'}
          </button>

          {queryMutation.isError && (
            <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg flex items-start gap-3">
              <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="text-red-400 font-medium">Error</p>
                <p className="text-red-300 text-sm mt-1">
                  {queryMutation.error?.message || 'Failed to process query'}
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="p-4 bg-slate-700 rounded-lg">
                <h4 className="text-white font-medium mb-3">Response</h4>
                <p className="text-slate-200 leading-relaxed whitespace-pre-wrap">
                  {result.data?.text || 'No response received'}
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-3 bg-slate-700 rounded-lg">
                  <p className="text-slate-400 text-xs mb-1">Model</p>
                  <p className="text-white font-medium text-sm">
                    {result.data?.model_used || 'N/A'}
                  </p>
                </div>
                <div className="p-3 bg-slate-700 rounded-lg">
                  <p className="text-slate-400 text-xs mb-1">Tokens</p>
                  <p className="text-white font-medium text-sm">
                    {result.data?.tokens_used || '0'}
                  </p>
                </div>
                <div className="p-3 bg-slate-700 rounded-lg">
                  <p className="text-slate-400 text-xs mb-1">Cost</p>
                  <p className="text-white font-medium text-sm">
                    ${result.data?.cost_usd?.toFixed(4) || '0.0000'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </form>
      </div>
    </div>
  )
}

export default LlmPage
