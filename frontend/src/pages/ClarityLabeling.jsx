import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { BrainCircuit, Download, FileSpreadsheet, Loader2, PauseCircle, CheckCircle2, AlertTriangle, Rows3, Sparkles } from 'lucide-react'

import CSVUpload from '../components/CSVUpload'
import { startClarityJob } from '../api/clarity'

const API_BASE = 'http://localhost:8000'

export default function ClarityLabeling() {
  const [csvFile, setCsvFile] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [jobMeta, setJobMeta] = useState({ totalPrompts: 0, totalChunks: 0, processedPrompts: 0 })
  const [chunks, setChunks] = useState([])
  const [done, setDone] = useState(false)
  const sourceRef = useRef(null)

  const handleStart = async () => {
      setError(null)
    setRunning(true)
    setDone(false)
    setJobId(null)
    setChunks([])
    setJobMeta({ totalPrompts: 0, totalChunks: 0, processedPrompts: 0 })

    try {
      const { job_id: jobId } = await startClarityJob({
        file: csvFile,
        prompt_complexity: 'mid',
        use_case: 'text-generation',
      })
      setJobId(jobId)
      const source = new EventSource(`${API_BASE}/api/clarity/stream/${jobId}`)
      sourceRef.current = source

      source.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.type === 'started') {
          setJobMeta({
            totalPrompts: data.total_prompts,
            totalChunks: data.total_chunks,
            processedPrompts: 0,
          })
        }

        if (data.type === 'chunk_ready') {
          setChunks((prev) => [...prev, data])
          setJobMeta((prev) => ({
            ...prev,
            processedPrompts: data.processed_prompts,
            totalPrompts: data.total_prompts,
          }))
        }
        if (data.type === 'done') {
          setDone(true)
          setRunning(false)
          setJobMeta({
            totalPrompts: data.total_prompts,
            totalChunks: data.total_chunks,
            processedPrompts: data.processed_prompts,
          })
          source.close()
        }

        if (data.type === 'error') {
          setError(data.message)
          setRunning(false)
          source.close()
        }
      }

      source.onerror = () => {
        setError('Clarity stream disconnected. Check backend connectivity.')
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

  const progressPercent = jobMeta.totalPrompts
    ? Math.round((jobMeta.processedPrompts / jobMeta.totalPrompts) * 100)
    : 0

  return (
    <div className="max-w-5xl mx-auto px-8 py-12 space-y-10">
      <motion.div
        initial={{ opacity: 0, y: -18 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-end justify-between gap-6"
      >
        <div className="space-y-4 max-w-3xl">
          <div className="flex items-center gap-2 px-3 py-1 bg-emerald-600/10 border border-emerald-500/20 rounded-full w-fit">
            <BrainCircuit className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-[10px] font-black uppercase text-emerald-400 tracking-widest">Prompt Clarity Pipeline</span>
          </div>
          <h1 className="text-4xl font-black tracking-tight text-white">
            Prompt <span className="gradient-text">Clarity</span> Labeler
          </h1>
          <p className="text-gray-400 font-medium leading-relaxed">
            Upload a prompt CSV, send 5 prompts at a time to GPT-4.1, and download each finished 5-row
            chunk as <span className="text-white font-semibold">prompt_set_N.csv</span> while the rest continue processing.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-3 min-w-[280px]">
          <div className="glass-card rounded-2xl p-4 border border-white/5">
            <div className="text-[10px] uppercase tracking-widest text-gray-500 font-black">API Batch</div>
            <div className="text-2xl font-black text-white mt-2">5</div>
          </div>
          <div className="glass-card rounded-2xl p-4 border border-white/5">
            <div className="text-[10px] uppercase tracking-widest text-gray-500 font-black">CSV Size</div>
            <div className="text-2xl font-black text-white mt-2">5</div>
          </div>
          <div className="glass-card rounded-2xl p-4 border border-white/5">
            <div className="text-[10px] uppercase tracking-widest text-gray-500 font-black">Model</div>
            <div className="text-lg font-black text-white mt-3">GPT-4.1</div>
          </div>
        </div>
      </motion.div>

      <div className="glass-card rounded-2xl p-8 border border-white/5 space-y-8">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-emerald-400" />
          <h2 className="text-sm font-semibold text-gray-300">Upload Prompt CSV</h2>
        </div>

        <CSVUpload
          file={csvFile}
          onFileChange={setCsvFile}
          title="Prompt Source CSV"
          idleText="Drag & drop prompt CSV for clarity labeling"
          helperText='CSV must include a "prompt" column. Each generated file will contain only "prompt" and "clarity".'
        />

        <div className="rounded-2xl bg-black/30 border border-white/5 p-5 space-y-3">
          <div className="flex items-center gap-2 text-white font-bold">
            <Rows3 className="w-4 h-4 text-emerald-400" />
            Processing Rules
          </div>
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div className="rounded-xl bg-white/5 border border-white/5 p-4 text-gray-300">
              Every OpenAI request contains exactly 5 prompts.
            </div>
            <div className="rounded-xl bg-white/5 border border-white/5 p-4 text-gray-300">
              Every downloadable CSV is written as a separate 5-row chunk.
            </div>
            <div className="rounded-xl bg-white/5 border border-white/5 p-4 text-gray-300">
              Output columns are only <span className="text-white font-semibold">prompt</span> and <span className="text-white font-semibold">clarity</span>.
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleStart}
            disabled={!csvFile || running}
            className="flex items-center gap-3 px-8 py-3.5 bg-emerald-600 text-white font-black text-xs uppercase tracking-widest rounded-xl hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed transition-all"
          >
            {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileSpreadsheet className="w-4 h-4" />}
            {running ? 'Labeling Prompts...' : 'Start Clarity Labeling'}
          </button>

          {running && (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-6 py-3.5 rounded-xl border border-red-500/20 text-red-400 bg-red-950/20 text-xs font-black uppercase tracking-widest"
            >
              <PauseCircle className="w-4 h-4" />
              Stop Stream
            </button>
          )}
        </div>
      </div>

      {(running || done || chunks.length > 0 || error) && (
        <div className="space-y-8">
          <div className="glass-card rounded-2xl p-8 border border-white/5 space-y-4">
            <div className="flex items-end justify-between gap-6">
              <div>
                <div className="text-[10px] uppercase tracking-widest text-emerald-400 font-black">Clarity Progress</div>
                <div className="text-3xl font-black text-white mt-2">
                  {jobMeta.processedPrompts} <span className="text-gray-600 text-lg">/ {jobMeta.totalPrompts || 0}</span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] uppercase tracking-widest text-gray-500 font-black">Downloadable Chunks</div>
                <div className="text-2xl font-black text-white mt-2">{chunks.length}</div>
              </div>
            </div>

            <div className="h-3 bg-gray-900/80 rounded-full overflow-hidden border border-white/5">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${progressPercent}%` }}
                className="h-full bg-gradient-to-r from-emerald-600 via-emerald-400 to-cyan-400"
              />
            </div>
          </div>

          <div className="glass-card rounded-2xl p-8 border border-white/5 space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h3 className="text-white text-lg font-black tracking-tight">Ready CSV Chunks</h3>
                <p className="text-sm text-gray-500">Each file contains one processed batch from GPT-4.1.</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-xs uppercase tracking-widest text-gray-500 font-black">
                  {jobMeta.totalChunks ? `${chunks.length} of ${jobMeta.totalChunks}` : 'Waiting'}
                </div>
                {jobId && chunks.length > 0 && (
                  <a
                    href={`${API_BASE}/api/clarity/download-zip/${jobId}`}
                    className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-600 text-white text-[10px] font-black uppercase tracking-widest hover:bg-cyan-500 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download ZIP
                  </a>
                )}
              </div>
            </div>

            <div className="grid gap-3">
              {chunks.length === 0 && (
                <div className="rounded-xl border border-dashed border-white/10 bg-black/20 p-6 text-gray-500 text-sm">
                  No chunks ready yet.
                </div>
              )}

              {chunks.map((chunk) => (
                <div
                  key={chunk.file_name}
                  className="rounded-2xl bg-black/25 border border-white/5 px-5 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4"
                >
                  <div>
                    <div className="text-white font-black tracking-tight">{chunk.file_name}</div>
                    <div className="text-sm text-gray-500 mt-1">
                      Chunk {chunk.chunk_index} • {chunk.chunk_size} prompts • processed {chunk.processed_prompts} of {chunk.total_prompts}
                    </div>
                  </div>

                  <a
                    href={`${API_BASE}${chunk.download_url}`}
                    className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-emerald-600 text-white text-xs font-black uppercase tracking-widest hover:bg-emerald-500 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </a>
                </div>
              ))}
            </div>
          </div>

          {done && (
            <div className="rounded-2xl border border-green-500/20 bg-green-950/20 px-6 py-5 flex items-center gap-4">
              <CheckCircle2 className="w-6 h-6 text-green-400" />
              <div>
                <div className="text-green-300 font-black">Clarity labeling finished</div>
                <div className="text-green-400/70 text-sm">
                  Processed {jobMeta.processedPrompts} prompts into {jobMeta.totalChunks} downloadable chunk files.
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-2xl border border-red-500/20 bg-red-950/20 px-6 py-5 flex items-center gap-4">
              <AlertTriangle className="w-6 h-6 text-red-400" />
              <div>
                <div className="text-red-300 font-black">Clarity labeling failed</div>
                <div className="text-red-400/70 text-sm">{error}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
