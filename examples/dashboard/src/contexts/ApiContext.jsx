import React, { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'

const ApiContext = createContext()

export const useApi = () => {
  const context = useContext(ApiContext)
  if (!context) {
    throw new Error('useApi must be used within ApiProvider')
  }
  return context
}

export const ApiProvider = ({ children }) => {
  const [apiKey, setApiKey] = useState(() => {
    return localStorage.getItem('prism_api_key') || ''
  })

  const [baseUrl, setBaseUrl] = useState(() => {
    return localStorage.getItem('prism_base_url') || 'http://localhost:8080'
  })

  useEffect(() => {
    localStorage.setItem('prism_api_key', apiKey)
  }, [apiKey])

  useEffect(() => {
    localStorage.setItem('prism_base_url', baseUrl)
  }, [baseUrl])

  const client = axios.create({
    baseURL: baseUrl,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    timeout: 30000,
  })

  const api = {
    // Health
    health: () => client.get('/health'),

    // PWSA
    detectThreat: (data) => client.post('/api/v1/pwsa/detect', data),
    fuseSensors: (data) => client.post('/api/v1/pwsa/fuse', data),
    predictTrajectory: (data) => client.post('/api/v1/pwsa/predict', data),
    prioritizeThreats: (data) => client.post('/api/v1/pwsa/prioritize', data),

    // Finance
    optimizePortfolio: (data) => client.post('/api/v1/finance/optimize', data),
    assessRisk: (data) => client.post('/api/v1/finance/risk', data),
    backtestStrategy: (data) => client.post('/api/v1/finance/backtest', data),

    // LLM
    queryLlm: (data) => client.post('/api/v1/llm/query', data),
    llmConsensus: (data) => client.post('/api/v1/llm/consensus', data),
    listModels: () => client.get('/api/v1/llm/models'),

    // Time Series
    forecastTimeSeries: (data) => client.post('/api/v1/timeseries/forecast', data),

    // Pixels
    processPixels: (data) => client.post('/api/v1/pixels/process', data),
  }

  const value = {
    apiKey,
    setApiKey,
    baseUrl,
    setBaseUrl,
    api,
    isConfigured: !!apiKey,
  }

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>
}
