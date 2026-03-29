import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Square, CheckCircle2, AlertTriangle, Cpu, LayoutGrid, FileText, ChevronRight, Activity, Gauge, XCircle, Code2, Brain, MessageSquareText as TextIcon, Sparkles } from 'lucide-react'
import PromptInput from '../components/PromptInput'
import CSVUpload from '../components/CSVUpload'
import LiveLog from '../components/LiveLog'
import { startTrainingJob, startCSVTrainingJob, startMultiCSVTrainingJob } from '../api/training'

const COMPLEXITY_OPTIONS = [
  { value: 'low',  label: 'Low',  desc: 'Simple factual, one-step',  color: 'green' },
  { value: 'mid',  label: 'Mid',  desc: 'Multi-step, moderate logic', color: 'yellow' },
  { value: 'high', label: 'High', desc: 'Expert, multi-constraint',   color: 'red' },
]

const USE_CASE_OPTIONS = [
  { value: 'text-generation', label: 'Text Generation', desc: 'Summarization, chat, content', icon: TextIcon, color: 'blue', models: 17 },
  { value: 'code-generation', label: 'Code Generation', desc: 'HumanEval, SWE-bench tasks',   icon: Code2,    color: 'purple', models: 14 },
  { value: 'reasoning',       label: 'Reasoning',       desc: 'AIME, GPQA, chain-of-thought', icon: Brain,    color: 'orange', models: 12 },
]

const CLARITY_OPTIONS = [
  { value: 'CLEAR',   label: 'Clear',   desc: 'Well-defined, unambiguous', color: 'green' },
  { value: 'PARTIAL', label: 'Partial', desc: 'Some ambiguity present',    color: 'yellow' },
  { value: 'UNCLEAR', label: 'Unclear', desc: 'Vague or underspecified',   color: 'red' },
]

export default function Training() {
  const [promptComplexity, setPromptComplexity] = useState('mid')
  const [useCase, setUseCase] = useState('text-generation')
  const [clarity, setClarity] = useState('CLEAR')
  const [inputMode, setInputMode] = useState('single') // 'single' | 'csv'
  const [prompt, setPrompt] = useState('')
  const [csvFile, setCsvFile] = useState(null)
  const [csvFiles, setCsvFiles] = useState([])
  const [csvDelayMs, setCsvDelayMs] = useState(3000)
  const [logs, setLogs] = useState([])
  const [failedLogs, setFailedLogs] = useState([])
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [fileProgress, setFileProgress] = useState(null)
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState(null)
  const sourceRef = useRef(null)

  const selectedUseCase = USE_CASE_OPTIONS.find(u => u.value === useCase)

  const handleRun = async () => {
    setLogs([])
    setFailedLogs([])
    setProgress({ current: 0, total: 0 })
    setFileProgress(null)
    setDone(false)
    setError(null)
    setRunning(true)

    try {
      let jobId
      if (inputMode === 'single') {
        const res = await startTrainingJob({
          prompt,
          prompt_complexity: promptComplexity,
          use_case: useCase,
          clarity,
        })
        jobId = res.job_id
      } else {
        const res = csvFiles.length > 1
          ? await startMultiCSVTrainingJob({
              files: csvFiles,
              prompt_complexity: promptComplexity,
              use_case: useCase,
              delay_ms: csvDelayMs,
            })
          : await startCSVTrainingJob({
              file: csvFile,
              prompt_complexity: promptComplexity,
              use_case: useCase,
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

        if (data.type === 'file_started') {
          setFileProgress({
            status: 'processing',
            fileIndex: data.file_index,
            totalFiles: data.total_files,
            fileName: data.file_name,
            processedPrompts: data.processed_prompts,
            totalPrompts: data.total,
          })
          setProgress({ current: data.processed_prompts, total: data.total })
        }

        if (data.type === 'file_done') {
          setFileProgress({
            status: 'completed',
            fileIndex: data.file_index,
            totalFiles: data.total_files,
            fileName: data.file_name,
            processedPrompts: data.processed_prompts,
            totalPrompts: data.total,
          })
          setProgress({ current: data.processed_prompts, total: data.total })
        }

        if (data.type === 'file_delay') {
          setFileProgress({
            status: 'waiting',
            fileIndex: data.file_index,
            totalFiles: csvFiles.length,
            fileName: null,
            processedPrompts: data.processed_prompts,
            totalPrompts: data.total,
            delayMs: data.delay_ms,
          })
          setProgress({ current: data.processed_prompts, total: data.total })
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
             Orchestrate massive parallel evaluation across 22 LLMs on AWS Bedrock and GCP Vertex. 
             Choose your use case and complexity, upload prompts with clarity labels, and let Gemini 2.5 Pro judge responses.
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
        {/* Row 1: Use Case Selector */}
        <div className="space-y-3">
           <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-5 h-5 text-blue-500" />
              <label className="text-sm font-semibold text-gray-300">
                Use Case
              </label>
           </div>
           
           <div className="grid grid-cols-3 gap-3">
             {USE_CASE_OPTIONS.map(opt => {
               const isSelected = useCase === opt.value
               const Icon = opt.icon
               const colorMap = {
                 blue:   { bg: 'bg-blue-500/10',   border: 'border-blue-500/30',   text: 'text-blue-400',   activeBg: 'bg-blue-500/20',   ring: 'ring-blue-500/20' },
                 purple: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400', activeBg: 'bg-purple-500/20', ring: 'ring-purple-500/20' },
                 orange: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', activeBg: 'bg-orange-500/20', ring: 'ring-orange-500/20' },
               }
               const c = colorMap[opt.color]
               
               return (
                 <button
                   key={opt.value}
                   onClick={() => setUseCase(opt.value)}
                   className={`relative flex flex-col items-center gap-2 px-4 py-5 rounded-xl border-2 transition-all duration-300 cursor-pointer
                     ${isSelected 
                       ? `${c.activeBg} ${c.border} ${c.text} ring-4 ${c.ring} shadow-lg` 
                       : 'bg-gray-950/40 border-gray-800/80 text-gray-500 hover:border-gray-600/80 hover:bg-gray-900/40'
                     }`}
                 >
                   {isSelected && (
                     <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${opt.color === 'blue' ? 'bg-blue-500' : opt.color === 'purple' ? 'bg-purple-500' : 'bg-orange-500'} border-2 border-gray-950 animate-pulse`} />
                   )}
                   <Icon className={`w-5 h-5 ${isSelected ? c.text : 'text-gray-600'}`} />
                   <span className={`text-sm font-black uppercase tracking-widest ${isSelected ? c.text : ''}`}>
                     {opt.label}
                   </span>
                   <span className={`text-[10px] font-medium ${isSelected ? 'opacity-80' : 'text-gray-600'}`}>
                     {opt.desc}
                   </span>
                   <span className={`text-[9px] font-mono font-bold ${isSelected ? 'opacity-70' : 'text-gray-700'}`}>
                     {opt.models} models
                   </span>
                 </button>
               )
             })}
           </div>
        </div>

        {/* Row 2: Complexity + Clarity */}
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

           {/* Clarity Selector (single mode only) */}
           {inputMode === 'single' && (
             <div className="lg:w-1/2 space-y-3">
                <div className="flex items-center gap-2 mb-3">
                   <Sparkles className="w-5 h-5 text-blue-500" />
                   <label className="text-sm font-semibold text-gray-300">
                     Prompt Clarity
                   </label>
                </div>
                
                <div className="grid grid-cols-3 gap-3">
                  {CLARITY_OPTIONS.map(opt => {
                    const isSelected = clarity === opt.value
                    const colorMap = {
                      green:  { bg: 'bg-green-500/10',  border: 'border-green-500/30',  text: 'text-green-400',  activeBg: 'bg-green-500/20', ring: 'ring-green-500/20' },
                      yellow: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', activeBg: 'bg-yellow-500/20', ring: 'ring-yellow-500/20' },
                      red:    { bg: 'bg-red-500/10',    border: 'border-red-500/30',    text: 'text-red-400',    activeBg: 'bg-red-500/20', ring: 'ring-red-500/20' },
                    }
                    const c = colorMap[opt.color]
                    
                    return (
                      <button
                        key={opt.value}
                        onClick={() => setClarity(opt.value)}
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
           )}
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
                   <div className="mt-5 space-y-4">
                     <CSVUpload
                       files={csvFiles}
                       onFileChange={(files) => {
                         setCsvFiles(files)
                         setCsvFile(files[0] || null)
                       }}
                       title="Sequential CSV Queue"
                       idleText="Optionally select multiple chunk CSVs to process one file at a time"
                       helperText='Each CSV must contain "prompt" and "clarity". Files will be processed sequentially with a pause between them.'
                       multiple
                     />
                     <div className="rounded-xl bg-black/20 border border-white/5 p-4 flex flex-col md:flex-row md:items-center gap-4">
                       <div className="text-sm text-gray-400 flex-1">
                         When multiple CSVs are selected, the backend will finish one file, wait, then start the next file.
                       </div>
                       <label className="flex items-center gap-3 text-sm font-semibold text-gray-300">
                         Delay Between Files
                         <input
                           type="number"
                           min="0"
                           step="500"
                           value={csvDelayMs}
                           onChange={(e) => setCsvDelayMs(Number(e.target.value) || 0)}
                           className="w-28 rounded-lg bg-gray-950/60 border border-gray-800 px-3 py-2 text-white"
                         />
                         <span className="text-gray-500 text-xs uppercase tracking-widest">ms</span>
                       </label>
                     </div>
                   </div>
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
                <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">Active Models</span>
                <span className="text-gray-400 font-black text-xs tracking-tight">{selectedUseCase?.models || 0} Models ({selectedUseCase?.label})</span>
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
          {fileProgress && (
            <div className="glass-card rounded-2xl p-6 border border-white/5">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-[10px] uppercase tracking-widest text-blue-500 font-black">
                    {fileProgress.status === 'waiting' ? 'Inter-File Delay' : 'CSV Queue'}
                  </div>
                  <div className="text-white font-black text-lg mt-2">
                    {fileProgress.status === 'processing' && `Processing ${fileProgress.fileName}`}
                    {fileProgress.status === 'completed' && `Completed ${fileProgress.fileName}`}
                    {fileProgress.status === 'waiting' && `Waiting ${Math.round((fileProgress.delayMs || 0) / 1000)}s before next CSV`}
                  </div>
                  <div className="text-sm text-gray-500 mt-1">
                    {fileProgress.fileIndex} of {fileProgress.totalFiles || csvFiles.length} file steps
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] uppercase tracking-widest text-gray-500 font-black">Overall Prompts</div>
                  <div className="text-2xl font-black text-white mt-2">
                    {fileProgress.processedPrompts} / {fileProgress.totalPrompts}
                  </div>
                </div>
              </div>
            </div>
          )}

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
