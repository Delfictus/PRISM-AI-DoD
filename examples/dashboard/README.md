# PRISM-AI Web Dashboard

A modern, real-time web dashboard for monitoring and interacting with the PRISM-AI REST API.

## Features

- **Real-time Monitoring**: Live API health status, uptime, and metrics
- **Interactive Controls**: Direct API interaction through intuitive UI
- **Multi-Domain Support**: PWSA, Finance, LLM operations
- **Beautiful Charts**: Visualize request rates and latency trends
- **Dark Mode UI**: Modern, responsive design with Tailwind CSS
- **Secure Configuration**: API key storage in browser localStorage

## Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **TailwindCSS** - Styling
- **React Router** - Navigation
- **TanStack Query** - Data fetching and caching
- **Recharts** - Data visualization
- **Lucide React** - Icons
- **Axios** - HTTP client

## Quick Start

### Prerequisites

- Node.js 16+ and npm
- PRISM-AI API server running (default: `http://localhost:8080`)

### Installation

```bash
cd examples/dashboard
npm install
```

### Development

```bash
npm run dev
```

Opens at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

Output in `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Configuration

### First Time Setup

1. Start the dashboard: `npm run dev`
2. Navigate to **Settings** page
3. Enter your API configuration:
   - **API Base URL**: `http://localhost:8080`
   - **API Key**: Your authentication key
4. Click **Save Configuration**
5. Go to **Dashboard** to verify connection

### Environment Variables

Create a `.env` file:

```env
VITE_API_BASE_URL=http://localhost:8080
```

## Pages

### Dashboard

- API health status
- System uptime
- Total requests counter
- Request rate charts
- Latency monitoring
- Recent activity feed

### PWSA - Threat Detection

- Detect threats from IR sensor data
- Input space vehicle ID
- Real-time threat analysis
- Display confidence scores and threat types

### Finance - Portfolio Optimization

- Optimize asset allocation
- Calculate Sharpe ratios
- View optimal portfolio weights
- Risk/return analysis

### LLM - Language Model

- Query language models
- Interactive chat interface
- Token usage tracking
- Cost estimation

### Settings

- Configure API base URL
- Set authentication key
- Connection testing
- Getting started guide

## API Integration

The dashboard integrates with all PRISM-AI API endpoints:

```javascript
// Example: Detect Threat
POST /api/v1/pwsa/detect
{
  "sv_id": 42,
  "timestamp": 1234567890,
  "ir_frame": {
    "width": 640,
    "height": 480,
    "centroid_x": 320.0,
    "centroid_y": 240.0,
    "hotspot_count": 5
  }
}

// Example: Optimize Portfolio
POST /api/v1/finance/optimize
{
  "assets": [
    { "symbol": "AAPL", "expected_return": 0.12, "volatility": 0.25 }
  ],
  "constraints": {
    "max_position_size": 0.5,
    "max_total_risk": 0.20
  }
}

// Example: Query LLM
POST /api/v1/llm/query
{
  "prompt": "Explain quantum computing",
  "temperature": 0.7,
  "max_tokens": 500
}
```

## Project Structure

```
dashboard/
├── public/                 # Static assets
├── src/
│   ├── components/        # Reusable components
│   │   └── Layout.jsx     # Navigation layout
│   ├── contexts/          # React contexts
│   │   └── ApiContext.jsx # API client context
│   ├── pages/             # Page components
│   │   ├── Dashboard.jsx  # Main dashboard
│   │   ├── PwsaPage.jsx   # PWSA operations
│   │   ├── FinancePage.jsx # Finance operations
│   │   ├── LlmPage.jsx    # LLM operations
│   │   └── SettingsPage.jsx # Configuration
│   ├── App.jsx            # Root component
│   ├── main.jsx           # Entry point
│   └── index.css          # Global styles
├── index.html             # HTML template
├── package.json           # Dependencies
├── vite.config.js         # Vite configuration
├── tailwind.config.js     # Tailwind configuration
└── README.md              # This file
```

## Development

### Adding a New Page

1. Create component in `src/pages/`
2. Add route in `src/App.jsx`
3. Add navigation link in `src/components/Layout.jsx`

### Adding New API Methods

1. Add method to `src/contexts/ApiContext.jsx`:

```javascript
const api = {
  // ... existing methods
  newMethod: (data) => client.post('/api/v1/new/endpoint', data),
}
```

2. Use in components with `useApi()` hook:

```javascript
const { api } = useApi()
const mutation = useMutation({
  mutationFn: (data) => api.newMethod(data),
})
```

### Styling

Dashboard uses Tailwind CSS utility classes:

```jsx
<div className="card">
  <button className="btn-primary">Click Me</button>
</div>
```

Custom components defined in `src/index.css`:
- `.card` - Card container
- `.btn-primary` - Primary button
- `.btn-secondary` - Secondary button
- `.input` - Form input
- `.badge-*` - Status badges

## API Proxy

Vite dev server proxies API requests:

```javascript
// vite.config.js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8080',
      changeOrigin: true,
    },
  },
}
```

This allows calling `/api/v1/pwsa/detect` which proxies to `http://localhost:8080/api/v1/pwsa/detect`.

## Authentication

API key stored in browser `localStorage`:

```javascript
localStorage.setItem('prism_api_key', 'your-key')
localStorage.setItem('prism_base_url', 'http://localhost:8080')
```

Sent in Authorization header:

```javascript
headers: {
  'Authorization': `Bearer ${apiKey}`,
}
```

## Error Handling

Errors displayed with user-friendly messages:

```javascript
{queryMutation.isError && (
  <div className="error-alert">
    {queryMutation.error?.message || 'Operation failed'}
  </div>
)}
```

## Real-time Updates

Dashboard polls health endpoint every 5 seconds:

```javascript
useQuery({
  queryKey: ['health'],
  queryFn: () => api.health(),
  refetchInterval: 5000,
})
```

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Production Deployment

### Static Hosting

```bash
npm run build
# Upload dist/ to hosting service
```

### Docker

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

### Nginx Configuration

```nginx
server {
  listen 80;
  root /usr/share/nginx/html;
  index index.html;

  location / {
    try_files $uri $uri/ /index.html;
  }

  location /api {
    proxy_pass http://prism-api:8080;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
  }
}
```

## Troubleshooting

### CORS Issues

If running API on different origin, enable CORS on API server:

```rust
// In API server
.layer(
    CorsLayer::new()
        .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([AUTHORIZATION, CONTENT_TYPE])
)
```

### Connection Refused

Verify API server is running:

```bash
curl http://localhost:8080/health
```

### API Key Not Working

Check API key in Settings page matches server configuration.

## License

MIT License

## Support

- Documentation: https://docs.prism-ai.example.com
- Issues: https://github.com/your-org/prism-ai/issues
- Email: support@prism-ai.example.com
