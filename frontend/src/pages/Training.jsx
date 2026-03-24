import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Square, CheckCircle2, AlertTriangle, Cpu, LayoutGrid, FileText, ChevronRight, Activity, Sparkles, Gauge, XCircle } from 'lucide-react'
import EvaluatorDropdown from '../components/EvaluatorDropdown'
import PromptInput from '../components/PromptInput'
import CSVUpload from '../components/CSVUpload'
import LiveLog from '../components/LiveLog'
import { startTrainingJob, startCSVTrainingJob } from '../api/training'

const COMPLEXITY_OPTIONS = [
  { value: 'low',  label: 'Low',  desc: 'Simple factual, one-step',  color: 'green' },
  { value: 'mid',  label: 'Mid',  desc: 'Multi-step, moderate logic', color: 'yellow' },
  { value: 'high', label: 'High', desc: 'Expert, multi-constraint',   color: 'red' },
]

export default function Training() {
  const [evaluatorModel, setEvaluatorModel] = useState('gemini-2.0-flash')
  const [promptComplexity, setPromptComplexity] = useState('mid')
  const [promptQualityScore, setPromptQualityScore] = useState(50)
  const [inputMode, setInputMode] = useState('single') // 'single' | 'csv'
  const [prompt, setPrompt] = useState('')
  const [csvFile, setCsvFile] = useState(null)
  const [logs, setLogs] = useState([])
  const [failedLogs, setFailedLogs] = useState([])
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState(null)
  const sourceRef = useRef(null)

  const handleRun = async () => {
    setLogs([])
    setFailedLogs([])
    setProgress({ current: 0, total: 0 })
    setDone(false)
    setError(null)
    setRunning(true)

    try {
      let jobId
      if (inputMode === 'single') {
        const res = await startTrainingJob({
          prompt,
          prompt_complexity: promptComplexity,
          prompt_quality_score: promptQualityScore,
          evaluator_model: evaluatorModel,
        })
        jobId = res.job_id
      } else {
        const res = await startCSVTrainingJob({
          file: csvFile,
          prompt_complexity: promptComplexity,
          evaluator_model: evaluatorModel,
        })
        jobId = res.job_id
      }

      const source = new EventSource(`http://localhost:8000/api/training/stream/${jobId}`)
      sourceRef.current = source

      source.onmessage = (e) => {
        const data = JSON.parse(e.data)

        if (data.type === 'progress') {
          setProgress({ current: data.prompt_index, total: data.total })
          setLogs(prev => [...prev, data])
        }

        if (data.type === 'model_failed') {
          setFailedLogs(prev => [...prev, data])
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
        setError('SSE Connection Lost. Verify backend reachability.')
        setRunning(false)
        source.close()
      }

    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setRunning(false)
    }
  }

  const handleStop = () => {
    sourceRef.current?.close()
    setRunning(false)
  }

  const totalResults = logs.length
  const complexityColor = promptComplexity === 'high' ? 'red' : promptComplexity === 'mid' ? 'yellow' : 'green'

  return (
    <div className="max-w-5xl mx-auto px-8 py-12 space-y-12">

      {/* Hero Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row md:items-end justify-between gap-6"
      >
        <div className="space-y-4 max-w-2xl">
           <div className="flex items-center gap-2 px-3 py-1 bg-blue-600/10 border border-blue-500/20 rounded-full w-fit">
              <Activity className="w-3.5 h-3.5 text-blue-500" />
              <span className="text-[10px] font-black uppercase text-blue-500 tracking-widest">Global Benchmarking</span>
           </div>
           <h1 className="text-4xl font-black text-white tracking-tight">
             Model <span className="gradient-text">Matrix</span> Benchmarks
           </h1>
           <p className="text-gray-400 font-medium leading-relaxed">
             Orchestrate massive parallel evaluation across 16+ LLMs on AWS Bedrock and GCP Vertex. 
             Choose your prompt complexity, upload prompts with accuracy scores, and let the AI Judge rate responses.
           </p>
        </div>
        
        <div className="flex items-center gap-3">
           <div className="h-12 w-[1px] bg-gray-800 hidden md:block mx-4" />
           <div className="flex flex-col items-center gap-1">
              <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest leading-none">Job Queue</span>
              <span className="text-white font-black text-2xl tracking-tighter leading-none">0/0</span>
           </div>
        </div>
      </motion.div>

      {/* Config Panel */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="glass-card rounded-2xl p-8 border border-white/5 space-y-8"
      >
        {/* Top Row: Complexity Selector + Evaluator Dropdown */}
        <div className="flex flex-col lg:flex-row gap-8">
           
           {/* Prompt Complexity Selector */}
           <div className="lg:w-1/2 space-y-3">
              <div className="flex items-center gap-2 mb-3">
                 <Gauge className="w-5 h-5 text-blue-500" />
                 <label className="text-sm font-semibold text-gray-300">
                   Prompt Complexity Level
                 </label>
              </div>
              
              <div className="grid grid-cols-3 gap-3">
                {COMPLEXITY_OPTIONS.map(opt => {
                  const isSelected = promptComplexity === opt.value
                  const colorMap = {
                    green:  { bg: 'bg-green-500/10',  border: 'border-green-500/30',  text: 'text-green-400',  activeBg: 'bg-green-500/20', ring: 'ring-green-500/20' },
                    yellow: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', activeBg: 'bg-yellow-500/20', ring: 'ring-yellow-500/20' },
                    red:    { bg: 'bg-red-500/10',    border: 'border-red-500/30',    text: 'text-red-400',    activeBg: 'bg-red-500/20', ring: 'ring-red-500/20' },
                  }
                  const c = colorMap[opt.color]
                  
                  return (
                    <button
                      key={opt.value}
                      onClick={() => setPromptComplexity(opt.value)}
                      className={`relative flex flex-col items-center gap-1.5 px-4 py-4 rounded-xl border-2 transition-all duration-300 cursor-pointer
                        ${isSelected 
                          ? `${c.activeBg} ${c.border} ${c.text} ring-4 ${c.ring} shadow-lg` 
                          : 'bg-gray-950/40 border-gray-800/80 text-gray-500 hover:border-gray-600/80 hover:bg-gray-900/40'
                        }`}
                    >
                      {isSelected && (
                        <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${c.text === 'text-green-400' ? 'bg-green-500' : c.text === 'text-yellow-400' ? 'bg-yellow-500' : 'bg-red-500'} border-2 border-gray-950 animate-pulse`} />
                      )}
                      <span className={`text-sm font-black uppercase tracking-widest ${isSelected ? c.text : ''}`}>
                        {opt.label}
                      </span>
                      <span className={`text-[10px] font-medium ${isSelected ? 'opacity-80' : 'text-gray-600'}`}>
                        {opt.desc}
                      </span>
                    </button>
                  )
                })}
              </div>
           </div>

           {/* Evaluator Dropdown */}
           <div className="lg:w-1/2">
              <EvaluatorDropdown value={evaluatorModel} onChange={setEvaluatorModel} />
           </div>
        </div>
           
        {/* Input Strategy */}
        <div className="space-y-6">
           <div className="flex items-center justify-between">
             <div className="flex items-center gap-2">
                <LayoutGrid className="w-4 h-4 text-blue-500" />
                <h3 className="text-sm font-bold text-gray-300 uppercase tracking-widest">Input Strategy</h3>
             </div>
             
             <div className="flex p-1 bg-black/40 rounded-xl border border-white/5 shadow-inner">
               {['single', 'csv'].map(mode => (
                 <button
                   key={mode}
                   onClick={() => setInputMode(mode)}
                   className={`px-5 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all duration-500 flex items-center gap-2 ${
                     inputMode === mode
                       ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                       : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                   }`}
                 >
                   {mode === 'single' ? <FileText className="w-3.5 h-3.5" /> : <Square className="w-3.5 h-3.5" />}
                   {mode} Mode
                 </button>
               ))}
             </div>
           </div>

           <div className="relative">
             <AnimatePresence mode="wait">
               {inputMode === 'single' ? (
                 <motion.div
                   key="single"
                   initial={{ opacity: 0, x: -10 }}
                   animate={{ opacity: 1, x: 0 }}
                   exit={{ opacity: 0, x: 10 }}
                   transition={{ duration: 0.3 }}
                   className="space-y-4"
                 >
                   <PromptInput value={prompt} onChange={setPrompt} />
                   
                   {/* Quality Score Slider for single mode */}
                   <div className="flex items-center gap-4 px-1">
                     <label className="text-xs font-bold text-gray-500 uppercase tracking-widest whitespace-nowrap">
                       Accuracy Score
                     </label>
                     <input
                       type="range"
                       min="0"
                       max="100"
                       value={promptQualityScore}
                       onChange={e => setPromptQualityScore(parseInt(e.target.value))}
                       className="flex-1 h-1.5 bg-gray-800 rounded-full appearance-none cursor-pointer
                                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 
                                  [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 
                                  [&::-webkit-slider-thumb]:shadow-lg [&::-webkit-slider-thumb]:shadow-blue-900/50
                                  [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:transition-all
                                  [&::-webkit-slider-thumb]:hover:scale-125"
                     />
                     <span className="text-blue-400 font-black text-sm tabular-nums w-10 text-right">
                       {promptQualityScore}
                     </span>
                   </div>
                 </motion.div>
               ) : (
                 <motion.div
                   key="csv"
                   initial={{ opacity: 0, x: 10 }}
                   animate={{ opacity: 1, x: 0 }}
                   exit={{ opacity: 0, x: -10 }}
                   transition={{ duration: 0.3 }}
                 >
                   <CSVUpload file={csvFile} onFileChange={setCsvFile} />
                 </motion.div>
               )}
             </AnimatePresence>
           </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-4 pt-4 border-t border-white/5">
          <button
            onClick={handleRun}
            disabled={running || (inputMode === 'single' ? !prompt.trim() : !csvFile)}
            className="flex items-center gap-3 px-8 py-3.5 bg-blue-600 text-white font-black text-xs uppercase tracking-widest rounded-xl
                       hover:bg-blue-500 active:scale-95 disabled:bg-gray-800 disabled:text-gray-600
                       disabled:cursor-not-allowed transition-all duration-300 shadow-xl shadow-blue-900/40 group"
          >
            {running ? (
               <>
                 <Cpu className="w-4 h-4 animate-spin-slow" />
                 Processing Matrix...
               </>
            ) : (
               <>
                 <Play className="w-4 h-4 transition-transform group-hover:scale-125 group-hover:fill-current" />
                 Execute Benchmark
               </>
            )}
          </button>
          
          {running && (
            <button
              onClick={handleStop}
              className="px-6 py-3.5 bg-gray-900 text-red-500 font-bold text-xs uppercase tracking-widest rounded-xl
                         border border-red-500/20 hover:bg-red-500/10 transition-all duration-300 flex items-center gap-2"
            >
              <Square className="w-4 h-4 fill-current" />
              Force Stop
            </button>
          )}

          <div className="flex-1" />
          
          <div className="hidden sm:flex items-center gap-6">
             <div className="flex flex-col items-end">
                <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Infrastructure</span>
                <span className="text-gray-400 font-black text-xs tracking-tight">16 Models (Global)</span>
             </div>
          </div>
        </div>
      </motion.div>

      {/* Live Status Overlay */}
      {(running || done || logs.length > 0 || failedLogs.length > 0) && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-8"
        >
          {/* Progress Visualizer */}
          {progress.total > 0 && (
            <div className="space-y-3 px-1">
              <div className="flex justify-between items-end">
                <div className="flex flex-col">
                  <span className="text-[10px] font-black text-blue-500 uppercase tracking-widest mb-1">Pipeline Stability: Stable</span>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-black text-white tracking-tighter">{progress.current}</span>
                    <span className="text-gray-600 font-bold text-sm">of {progress.total} prompts ingested</span>
                  </div>
                </div>
                <div className="flex items-center gap-1 text-[10px] font-black text-gray-500 uppercase tracking-widest">
                  {Math.round((progress.current / progress.total) * 100)}% Complete
                  <ChevronRight className="w-3 h-3 text-blue-600" />
                </div>
              </div>
              <div className="h-2.5 bg-gray-900/60 rounded-full border border-white/5 overflow-hidden shadow-inner group">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(progress.current / progress.total) * 100}%` }}
                  className="h-full bg-gradient-to-r from-blue-700 via-blue-500 to-indigo-500 relative shadow-2xl shadow-blue-500/50"
                >
                   <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.2)_50%,transparent_75%)] bg-[length:40px_40px] animate-[slide_2s_linear_infinite]" />
                </motion.div>
              </div>
            </div>
          )}

          <LiveLog logs={logs} />

          {/* Failed Models Warning */}
          {failedLogs.length > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-orange-950/20 border border-orange-500/20 rounded-2xl px-8 py-6 backdrop-blur-3xl shadow-2xl"
            >
              <div className="flex items-center gap-4 mb-4">
                <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center border border-orange-500/20">
                  <XCircle className="w-5 h-5 text-orange-400" />
                </div>
                <div>
                  <h4 className="text-orange-300 font-black tracking-tight">
                    {failedLogs.length} Model{failedLogs.length > 1 ? 's' : ''} Failed
                  </h4>
                  <p className="text-orange-400/60 text-xs font-medium">
                    These models returned null/empty responses and were excluded from scoring
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {failedLogs.map((fl, idx) => (
                  <span 
                    key={idx}
                    className="px-3 py-1.5 bg-orange-500/10 border border-orange-500/20 rounded-lg text-orange-400 text-[10px] font-black uppercase tracking-widest"
                  >
                    {fl.model_id}
                  </span>
                ))}
              </div>
            </motion.div>
          )}

          {/* Toast State Notifications */}
          <AnimatePresence>
            {done && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center justify-between bg-green-950/20 border border-green-500/20
                          rounded-2xl px-8 py-6 backdrop-blur-3xl shadow-2xl shadow-green-900/10"
              >
                <div className="flex items-center gap-5">
                   <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center border border-green-500/20">
                      <CheckCircle2 className="w-7 h-7 text-green-400" />
                   </div>
                   <div className="flex flex-col">
                      <h4 className="text-green-300 font-black tracking-tight text-lg">Benchmark Success</h4>
                      <p className="text-green-400/60 text-sm font-medium">
                        Successfully processed {totalResults} model outputs and synced to Supabase cluster.
                        {failedLogs.length > 0 && ` (${failedLogs.length} model failures excluded)`}
                      </p>
                   </div>
                </div>
                <button className="px-6 py-2.5 bg-green-500 text-gray-950 font-black text-[10px] uppercase tracking-widest rounded-lg hover:bg-green-400 transition-colors">
                  View Results
                </button>
              </motion.div>
            )}

            {error && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-5 bg-red-950/20 border border-red-500/20
                          rounded-2xl px-8 py-6 backdrop-blur-3xl shadow-2xl shadow-red-900/10"
              >
                <div className="w-12 h-12 rounded-xl bg-red-500/10 flex items-center justify-center border border-red-500/20">
                   <AlertTriangle className="w-7 h-7 text-red-500" />
                </div>
                <div className="flex flex-col">
                   <h4 className="text-red-300 font-black tracking-tight text-lg">Pipeline Error</h4>
                   <p className="text-red-400/60 text-sm font-medium">{error}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

    </div>
  )
}
