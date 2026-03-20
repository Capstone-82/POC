# Frontend Specification
## React + Vite — Dark Mode

---

## Setup

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install react-router-dom axios react-dropzone
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

---

## Folder Structure

```
frontend/
  src/
    pages/
      Training.jsx
      Inference.jsx
    components/
      Navbar.jsx
      EvaluatorDropdown.jsx
      PromptInput.jsx
      CSVUpload.jsx
      LiveLog.jsx
      RecommendationOutput.jsx
    api/
      training.js
      inference.js
    App.jsx
    main.jsx
    index.css
```

---

## Routing (App.jsx)

```jsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Training from './pages/Training'
import Inference from './pages/Inference'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-white">
        <Navbar />
        <Routes>
          <Route path="/" element={<Navigate to="/training" />} />
          <Route path="/training" element={<Training />} />
          <Route path="/inference" element={<Inference />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
```

---

## Navbar Component

```jsx
// src/components/Navbar.jsx
import { NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="border-b border-gray-800 bg-gray-950 px-8 py-4 flex items-center gap-8">
      <span className="text-white font-semibold text-lg tracking-tight">
        LLM Recommender
      </span>
      <div className="flex gap-2">
        <NavLink
          to="/training"
          className={({ isActive }) =>
            `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive
                ? 'bg-gray-800 text-white'
                : 'text-gray-400 hover:text-white'
            }`
          }
        >
          Training
        </NavLink>
        <NavLink
          to="/inference"
          className={({ isActive }) =>
            `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive
                ? 'bg-gray-800 text-white'
                : 'text-gray-400 hover:text-white'
            }`
          }
        >
          Inference
        </NavLink>
      </div>
    </nav>
  )
}
```

---

## Page 1 — Training.jsx

### Full Component

```jsx
// src/pages/Training.jsx
import { useState, useRef, useEffect } from 'react'
import EvaluatorDropdown from '../components/EvaluatorDropdown'
import PromptInput from '../components/PromptInput'
import CSVUpload from '../components/CSVUpload'
import LiveLog from '../components/LiveLog'
import { startTrainingJob, startCSVTrainingJob } from '../api/training'

export default function Training() {
  const [evaluatorModel, setEvaluatorModel] = useState('gemini-2.0-flash')
  const [inputMode, setInputMode] = useState('single') // 'single' | 'csv'
  const [prompt, setPrompt] = useState('')
  const [csvFile, setCsvFile] = useState(null)
  const [logs, setLogs] = useState([])
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState(null)
  const sourceRef = useRef(null)

  const handleRun = async () => {
    setLogs([])
    setProgress({ current: 0, total: 0 })
    setDone(false)
    setError(null)
    setRunning(true)

    try {
      // Step 1: Start job, get job_id
      let jobId
      if (inputMode === 'single') {
        const res = await startTrainingJob({ prompt, evaluator_model: evaluatorModel })
        jobId = res.job_id
      } else {
        const res = await startCSVTrainingJob({ file: csvFile, evaluator_model: evaluatorModel })
        jobId = res.job_id
      }

      // Step 2: Open SSE stream
      const source = new EventSource(`http://localhost:8000/api/training/stream/${jobId}`)
      sourceRef.current = source

      source.onmessage = (e) => {
        const data = JSON.parse(e.data)

        if (data.type === 'progress') {
          setProgress({ current: data.prompt_index, total: data.total })
          setLogs(prev => [...prev, data])
        }

        if (data.type === 'done') {
          setDone(true)
          setRunning(false)
          source.close()
        }

        if (data.type === 'error') {
          setError(data.message)
          setRunning(false)
          source.close()
        }
      }

      source.onerror = () => {
        setError('Connection lost. Check backend.')
        setRunning(false)
        source.close()
      }

    } catch (err) {
      setError(err.message)
      setRunning(false)
    }
  }

  const handleStop = () => {
    sourceRef.current?.close()
    setRunning(false)
  }

  const totalRows = logs.length

  return (
    <div className="max-w-4xl mx-auto px-8 py-10 space-y-8">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-white">Training</h1>
        <p className="text-gray-400 text-sm mt-1">
          Run prompts across all Bedrock models and store benchmark results in Supabase.
        </p>
      </div>

      {/* Config */}
      <div className="space-y-4">
        <EvaluatorDropdown value={evaluatorModel} onChange={setEvaluatorModel} />
      </div>

      {/* Input mode toggle */}
      <div>
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1 w-fit">
          {['single', 'csv'].map(mode => (
            <button
              key={mode}
              onClick={() => setInputMode(mode)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                inputMode === mode
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {mode === 'single' ? 'Single Prompt' : 'Upload CSV'}
            </button>
          ))}
        </div>

        <div className="mt-4">
          {inputMode === 'single' ? (
            <PromptInput value={prompt} onChange={setPrompt} />
          ) : (
            <CSVUpload file={csvFile} onFileChange={setCsvFile} />
          )}
        </div>
      </div>

      {/* Run button */}
      <div className="flex gap-3">
        <button
          onClick={handleRun}
          disabled={running || (inputMode === 'single' ? !prompt.trim() : !csvFile)}
          className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                     disabled:text-gray-500 text-white text-sm font-medium rounded-lg
                     transition-colors"
        >
          {running ? 'Running...' : 'Run Training'}
        </button>
        {running && (
          <button
            onClick={handleStop}
            className="px-6 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300
                       text-sm font-medium rounded-lg transition-colors"
          >
            Stop
          </button>
        )}
      </div>

      {/* Progress bar */}
      {(running || done) && progress.total > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-400">
            <span>Prompts processed</span>
            <span>{progress.current} / {progress.total}</span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Live log */}
      {logs.length > 0 && <LiveLog logs={logs} />}

      {/* Success */}
      {done && (
        <div className="flex items-center gap-3 bg-green-950 border border-green-800
                        rounded-lg px-5 py-4">
          <span className="text-green-400 text-lg">✓</span>
          <span className="text-green-300 text-sm">
            Training complete. {totalRows} rows saved to Supabase.
          </span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 bg-red-950 border border-red-800
                        rounded-lg px-5 py-4">
          <span className="text-red-400 text-lg">✗</span>
          <span className="text-red-300 text-sm">{error}</span>
        </div>
      )}

    </div>
  )
}
```

---

## Page 2 — Inference.jsx

```jsx
// src/pages/Inference.jsx
import { useState } from 'react'
import RecommendationOutput from '../components/RecommendationOutput'
import { getRecommendation } from '../api/inference'

const USE_CASES = [
  'Chat', 'Code', 'Reasoning', 'RAG', 'Summarization',
  'Structured Output', 'Tool Calling', 'Vision', 'Multimodality'
]

const MODELS = [
  'gpt-4o', 'gpt-4o-mini',
  'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
  'claude-sonnet-4-6',
  'deepseek-r1', 'deepseek-v3',
  'llama-3.3-70b', 'llama-4-scout',
  'mistral-large', 'mistral-small', 'pixtral-large',
  'nova-pro', 'nova-lite'
]

export default function Inference() {
  const [prompt, setPrompt] = useState('')
  const [useCase, setUseCase] = useState('')
  const [currentModel, setCurrentModel] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await getRecommendation({ prompt, use_case: useCase, current_model: currentModel })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = prompt.trim() && useCase && currentModel

  return (
    <div className="max-w-3xl mx-auto px-8 py-10 space-y-8">

      <div>
        <h1 className="text-2xl font-semibold text-white">Inference</h1>
        <p className="text-gray-400 text-sm mt-1">
          Get a model recommendation based on your prompt and benchmark data.
        </p>
      </div>

      <div className="space-y-4">
        {/* Prompt */}
        <div>
          <label className="block text-sm text-gray-400 mb-2">Your prompt</label>
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            rows={5}
            placeholder="Enter your prompt here..."
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                       text-white text-sm placeholder-gray-600 resize-none
                       focus:outline-none focus:border-gray-500 transition-colors"
          />
        </div>

        {/* Use case */}
        <div>
          <label className="block text-sm text-gray-400 mb-2">Use case</label>
          <select
            value={useCase}
            onChange={e => setUseCase(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2.5
                       text-white text-sm focus:outline-none focus:border-gray-500
                       transition-colors appearance-none"
          >
            <option value="">Select use case...</option>
            {USE_CASES.map(u => (
              <option key={u} value={u.toLowerCase()}>{u}</option>
            ))}
          </select>
        </div>

        {/* Current model */}
        <div>
          <label className="block text-sm text-gray-400 mb-2">Your current model</label>
          <select
            value={currentModel}
            onChange={e => setCurrentModel(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2.5
                       text-white text-sm focus:outline-none focus:border-gray-500
                       transition-colors appearance-none"
          >
            <option value="">Select model...</option>
            {MODELS.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={!canSubmit || loading}
        className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                   disabled:text-gray-500 text-white text-sm font-medium rounded-lg
                   transition-colors"
      >
        {loading ? 'Analyzing...' : 'Get Recommendation'}
      </button>

      {loading && (
        <div className="flex items-center gap-3 text-gray-400 text-sm">
          <div className="w-4 h-4 border-2 border-gray-600 border-t-blue-400
                          rounded-full animate-spin" />
          Running classifiers and recommendation model...
        </div>
      )}

      {result && <RecommendationOutput data={result} />}

      {error && (
        <div className="bg-red-950 border border-red-800 rounded-lg px-5 py-4
                        text-red-300 text-sm">
          {error}
        </div>
      )}

    </div>
  )
}
```

---

## Components

### EvaluatorDropdown.jsx

```jsx
const EVALUATORS = [
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash (default)' },
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
]

export default function EvaluatorDropdown({ value, onChange }) {
  return (
    <div>
      <label className="block text-sm text-gray-400 mb-2">Evaluator model</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full max-w-xs bg-gray-900 border border-gray-700 rounded-lg
                   px-4 py-2.5 text-white text-sm focus:outline-none
                   focus:border-gray-500 transition-colors appearance-none"
      >
        {EVALUATORS.map(e => (
          <option key={e.value} value={e.value}>{e.label}</option>
        ))}
      </select>
    </div>
  )
}
```

### PromptInput.jsx

```jsx
export default function PromptInput({ value, onChange }) {
  return (
    <div>
      <label className="block text-sm text-gray-400 mb-2">Prompt</label>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        rows={6}
        placeholder="Enter your prompt here..."
        className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-white text-sm placeholder-gray-600 resize-none
                   focus:outline-none focus:border-gray-500 transition-colors"
      />
    </div>
  )
}
```

### CSVUpload.jsx

```jsx
import { useDropzone } from 'react-dropzone'

export default function CSVUpload({ file, onFileChange }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    onDrop: files => onFileChange(files[0])
  })

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg px-8 py-10 text-center
                    cursor-pointer transition-colors ${
                      isDragActive
                        ? 'border-blue-500 bg-blue-950/20'
                        : 'border-gray-700 hover:border-gray-600'
                    }`}
      >
        <input {...getInputProps()} />
        {file ? (
          <div className="space-y-1">
            <p className="text-white text-sm font-medium">{file.name}</p>
            <p className="text-gray-500 text-xs">Click or drag to replace</p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="text-gray-300 text-sm">
              {isDragActive ? 'Drop CSV here' : 'Drag & drop CSV or click to browse'}
            </p>
            <p className="text-gray-600 text-xs">Must have a column named "prompt"</p>
          </div>
        )}
      </div>
    </div>
  )
}
```

### LiveLog.jsx

```jsx
import { useEffect, useRef } from 'react'

export default function LiveLog({ logs }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div>
      <label className="block text-sm text-gray-400 mb-2">Live output</label>
      <div className="bg-gray-950 border border-gray-800 rounded-lg p-4
                      h-80 overflow-y-auto font-mono text-xs space-y-1">
        {logs.map((log, i) => (
          <div key={i} className="flex gap-3 text-gray-300">
            <span className="text-gray-600 shrink-0">
              [{String(log.prompt_index).padStart(2, '0')}/{log.total}]
            </span>
            <span className="text-green-400 shrink-0">✓</span>
            <span className="text-blue-400 w-36 shrink-0">{log.model_id}</span>
            <span className="text-gray-400">
              accuracy: <span className="text-white">{log.accuracy_score}</span>
            </span>
            <span className="text-gray-400">
              complexity: <span className="text-yellow-400">{log.prompt_complexity}</span>
            </span>
            <span className="text-gray-400">
              cost: <span className="text-white">${log.cost?.toFixed(4)}</span>
            </span>
            <span className="text-gray-400">
              latency: <span className="text-white">{log.latency_ms}ms</span>
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
```

### RecommendationOutput.jsx

```jsx
export default function RecommendationOutput({ data }) {
  const fmt = (val, unit = '') => {
    const sign = val > 0 ? '+' : ''
    const color = val > 0 ? 'text-green-400' : 'text-red-400'
    return <span className={color}>{sign}{val}{unit}</span>
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-5">

      {/* Prompt meta */}
      <div className="flex gap-4 text-sm text-gray-400">
        <span>
          Complexity:
          <span className="text-yellow-400 ml-1 font-medium">{data.complexity}</span>
        </span>
        <span>•</span>
        <span>
          Prompt quality:
          <span className="text-white ml-1 font-medium">{data.quality_score}/100</span>
        </span>
      </div>

      <div className="border-t border-gray-800" />

      {/* Models */}
      <div className="space-y-1 text-sm">
        <div className="text-gray-400">
          You chose:
          <span className="text-white ml-2 font-medium">{data.current_model}</span>
        </div>
        <div className="text-gray-400">
          Recommended:
          <span className="text-blue-400 ml-2 font-semibold text-base">
            {data.recommended_model}
          </span>
        </div>
      </div>

      {/* Deltas */}
      <div className="space-y-2 font-mono text-sm">
        <div>{fmt(data.accuracy_delta, '% accuracy')}</div>
        <div>{fmt(data.cost_delta, '% cost')}</div>
        <div>{fmt(data.latency_delta, 'ms latency')}</div>
      </div>

      <div className="border-t border-gray-800" />

      {/* Reason */}
      <p className="text-gray-400 text-sm leading-relaxed">
        <span className="text-gray-500">Reason: </span>
        {data.reason}
      </p>

    </div>
  )
}
```

---

## API Layer

### src/api/training.js

```js
import axios from 'axios'

const BASE = 'http://localhost:8000/api'

export async function startTrainingJob({ prompt, evaluator_model }) {
  const res = await axios.post(`${BASE}/training/run`, { prompt, evaluator_model })
  return res.data // { job_id }
}

export async function startCSVTrainingJob({ file, evaluator_model }) {
  const form = new FormData()
  form.append('file', file)
  form.append('evaluator_model', evaluator_model)
  const res = await axios.post(`${BASE}/training/upload`, form)
  return res.data // { job_id }
}
```

### src/api/inference.js

```js
import axios from 'axios'

const BASE = 'http://localhost:8000/api'

export async function getRecommendation({ prompt, use_case, current_model }) {
  const res = await axios.post(`${BASE}/inference/recommend`, {
    prompt,
    use_case,
    current_model
  })
  return res.data
}
```

---

## Tailwind Config

```js
// tailwind.config.js
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: { extend: {} },
  plugins: [],
}
```

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  background-color: #030712;
}
```
