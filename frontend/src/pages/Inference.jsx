import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Zap,
  MessageSquare,
  Briefcase,
  Cpu,
  Loader2,
  Sparkles,
  AlertCircle,
  History,
  Info,
  Gauge,
  Coins,
  TimerReset,
  Database,
  ChevronDown,
} from 'lucide-react'
import RecommendationOutput from '../components/RecommendationOutput'
import { getRecommendation, getRecommendationOptions } from '../api/inference'

const fmt = (value, digits = 2) => {
  if (typeof value !== 'number') return 'N/A'
  return value.toFixed(digits)
}

export default function Inference() {
  const [prompt, setPrompt] = useState('')
  const [useCase, setUseCase] = useState('')
  const [currentModel, setCurrentModel] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [optionError, setOptionError] = useState(null)
  const [catalog, setCatalog] = useState({ use_cases: [], models: [], data_source: 'loading' })

  useEffect(() => {
    let active = true

    const loadOptions = async () => {
      try {
        const data = await getRecommendationOptions()
        if (!active) return
        setCatalog(data)
        if (!useCase && data.use_cases?.length) {
          setUseCase(data.use_cases[0].value)
        }
      } catch (err) {
        if (!active) return
        setOptionError(err.response?.data?.detail || err.message)
      }
    }

    loadOptions()
    return () => {
      active = false
    }
  }, [])

  const filteredModels = useMemo(() => {
    if (!useCase) return catalog.models || []
    return (catalog.models || []).filter((model) => model.use_cases?.includes(useCase))
  }, [catalog.models, useCase])

  useEffect(() => {
    if (!filteredModels.length) {
      setCurrentModel('')
      return
    }
    const stillExists = filteredModels.some((model) => model.model_id === currentModel)
    if (!stillExists) {
      setCurrentModel(filteredModels[0].model_id)
    }
  }, [filteredModels, currentModel])

  const selectedUseCase = useMemo(
    () => (catalog.use_cases || []).find((item) => item.value === useCase),
    [catalog.use_cases, useCase]
  )

  const selectedModel = useMemo(
    () => filteredModels.find((item) => item.model_id === currentModel),
    [filteredModels, currentModel]
  )

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await getRecommendation({
        prompt,
        use_case: useCase,
        current_model: currentModel
      })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = prompt.trim() && useCase && currentModel

  return (
    <div className="max-w-7xl mx-auto px-6 lg:px-10 py-10 lg:py-14 space-y-10 pb-28">
      <motion.div
        initial={{ opacity: 0, y: -18 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-[28px] border border-white/8 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.16),transparent_32%),radial-gradient(circle_at_80%_20%,rgba(59,130,246,0.18),transparent_28%),linear-gradient(180deg,rgba(10,18,35,0.96),rgba(6,12,26,0.96))] p-8 lg:p-10 shadow-[0_30px_120px_rgba(2,8,23,0.55)]"
      >
        <div className="absolute inset-y-0 right-0 w-[34%] bg-[linear-gradient(180deg,rgba(8,15,31,0),rgba(34,211,238,0.08),rgba(8,15,31,0))] pointer-events-none" />
        <div className="relative grid gap-8 lg:grid-cols-[1.35fr_0.85fr] lg:items-end">
          <div className="space-y-5">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-400/20 rounded-full w-fit">
              <Zap className="w-3.5 h-3.5 text-cyan-300" />
              <span className="text-[10px] font-black uppercase text-cyan-300 tracking-[0.24em]">Inference Optimizer</span>
            </div>
            <div className="space-y-3">
              <h1 className="max-w-3xl text-4xl lg:text-6xl font-black text-white tracking-tight leading-[0.95]">
                Model routing that feels <span className="gradient-text">decisive</span>, not generic.
              </h1>
              <p className="max-w-2xl text-base lg:text-lg text-slate-300/80 font-medium leading-relaxed">
                Pick a use case, compare real baseline models, and get a recommendation shaped by matched benchmark slices instead of a blunt global average.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Use Cases</p>
              <p className="mt-2 text-2xl font-black text-white">{catalog.use_cases?.length || 0}</p>
              <p className="text-xs text-slate-400 mt-1">Curated routing tracks</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Models</p>
              <p className="mt-2 text-2xl font-black text-white">{catalog.models?.length || 0}</p>
              <p className="text-xs text-slate-400 mt-1">Benchmark-backed baselines</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Data Source</p>
              <p className="mt-2 text-lg font-black text-white">{catalog.data_source || 'loading'}</p>
              <p className="text-xs text-slate-400 mt-1">Active recommendation source</p>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="grid gap-8 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-8">
          <div className="glass-card rounded-[28px] p-7 lg:p-8 border border-white/8 space-y-8">
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-cyan-300" />
                <label className="text-xs font-black text-slate-400 uppercase tracking-[0.24em]">Target Prompt</label>
              </div>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={7}
                placeholder="Paste the prompt you want to route. The system will infer complexity, match benchmark slices, and compare the current baseline against safer alternatives."
                className="w-full rounded-2xl border border-white/10 bg-slate-950/60 px-5 py-5 text-white text-sm placeholder:text-slate-600 resize-none focus:outline-none focus:border-cyan-400/40 focus:ring-4 focus:ring-cyan-950/30 transition-all"
              />
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-2">
                  <Briefcase className="w-4 h-4 text-cyan-300" />
                  <label className="text-xs font-black text-slate-400 uppercase tracking-[0.24em]">Use Case</label>
                </div>
                {selectedUseCase && (
                  <p className="text-xs text-slate-400 font-medium">{selectedUseCase.description}</p>
                )}
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                {(catalog.use_cases || []).map((item) => {
                  const selected = item.value === useCase
                  return (
                    <button
                      key={item.value}
                      type="button"
                      onClick={() => setUseCase(item.value)}
                      className={`rounded-2xl border p-4 text-left transition-all ${
                        selected
                          ? 'border-cyan-400/40 bg-cyan-400/10 shadow-[0_0_0_1px_rgba(34,211,238,0.12)]'
                          : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.06]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className={`text-sm font-black ${selected ? 'text-white' : 'text-slate-300'}`}>{item.label}</span>
                        <span className={`h-2.5 w-2.5 rounded-full ${selected ? 'bg-cyan-300' : 'bg-slate-600'}`} />
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-slate-400">{item.description}</p>
                    </button>
                  )
                })}
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-2">
                  <History className="w-4 h-4 text-cyan-300" />
                  <label className="text-xs font-black text-slate-400 uppercase tracking-[0.24em]">Baseline Model</label>
                </div>
                <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5">
                  <Database className="w-3.5 h-3.5 text-slate-400" />
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">
                    {filteredModels.length} compatible models
                  </span>
                  </div>
                </div>

              <div className="space-y-3">
                <div className="relative">
                  <select
                    value={currentModel}
                    onChange={(e) => setCurrentModel(e.target.value)}
                    className="w-full appearance-none rounded-2xl border border-white/10 bg-slate-950/60 px-5 py-4 pr-14 text-sm font-semibold text-white focus:outline-none focus:border-cyan-400/40 focus:ring-4 focus:ring-cyan-950/30 transition-all"
                  >
                    {filteredModels.map((model) => (
                      <option key={model.model_id} value={model.model_id} className="bg-slate-950 text-white">
                        {model.provider}/{model.model_id} | acc {fmt(model.avg_accuracy)} | ${fmt(model.median_cost, 6)} | {fmt(model.median_latency_ms, 0)}ms
                      </option>
                    ))}
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-4 flex items-center">
                    <ChevronDown className="w-4 h-4 text-slate-500" />
                  </div>
                </div>

                {selectedModel && (
                  <div className="rounded-2xl border border-blue-400/15 bg-blue-400/[0.05] px-4 py-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-base font-black text-white leading-tight break-words">
                          {selectedModel.provider}/{selectedModel.model_id}
                        </p>
                        <p className="mt-1 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                          Selected baseline | n={selectedModel.sample_count}
                        </p>
                      </div>
                      <span className="shrink-0 rounded-full bg-blue-400/20 px-2.5 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-blue-200">
                        Selected
                      </span>
                    </div>

                    <div className="mt-4 grid grid-cols-3 gap-2">
                      <div className="rounded-xl border border-white/8 bg-slate-950/50 px-3 py-2 min-w-0">
                        <div className="flex items-center gap-1.5 text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">
                          <Gauge className="w-3 h-3" />
                          Acc
                        </div>
                        <p className="mt-1 text-sm font-black text-white truncate">{fmt(selectedModel.avg_accuracy)}</p>
                      </div>
                      <div className="rounded-xl border border-white/8 bg-slate-950/50 px-3 py-2 min-w-0">
                        <div className="flex items-center gap-1.5 text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">
                          <Coins className="w-3 h-3" />
                          Cost
                        </div>
                        <p className="mt-1 text-sm font-black text-white truncate">${fmt(selectedModel.median_cost, 6)}</p>
                      </div>
                      <div className="rounded-xl border border-white/8 bg-slate-950/50 px-3 py-2 min-w-0">
                        <div className="flex items-center gap-1.5 text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">
                          <TimerReset className="w-3 h-3" />
                          Lat
                        </div>
                        <p className="mt-1 text-sm font-black text-white truncate">{fmt(selectedModel.median_latency_ms, 0)}ms</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={!canSubmit || loading}
              className="w-full rounded-2xl bg-[linear-gradient(90deg,#2563eb,#06b6d4)] px-6 py-4 text-xs font-black uppercase tracking-[0.24em] text-white transition-all shadow-[0_20px_50px_rgba(8,47,73,0.45)] hover:translate-y-[-1px] disabled:bg-slate-800 disabled:text-slate-500 disabled:shadow-none flex items-center justify-center gap-3"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running Recommendation Policy
                </>
              ) : (
                <>
                  Synthesize Recommendation
                  <Sparkles className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-[24px] border border-white/8 bg-[linear-gradient(180deg,rgba(15,23,42,0.9),rgba(13,20,35,0.88))] px-5 py-5">
            <div className="flex items-start gap-3">
              <div className="rounded-xl border border-cyan-400/20 bg-cyan-400/10 p-2.5">
                <Info className="w-4 h-4 text-cyan-300" />
              </div>
              <div>
                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Routing Notes</p>
                <p className="mt-2 text-sm text-slate-300 leading-relaxed">
                  Model cards show benchmark-backed averages so you can choose a realistic baseline before asking for a recommendation.
                </p>
                {selectedModel && (
                  <div className="mt-4 grid grid-cols-3 gap-2">
                    <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-3">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">Accuracy</p>
                      <p className="mt-1 text-sm font-black text-white">{fmt(selectedModel.avg_accuracy)}</p>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-3">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">Median Cost</p>
                      <p className="mt-1 text-sm font-black text-white">${fmt(selectedModel.median_cost, 6)}</p>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-3">
                      <p className="text-[10px] font-black uppercase tracking-[0.18em] text-slate-500">Median Latency</p>
                      <p className="mt-1 text-sm font-black text-white">{fmt(selectedModel.median_latency_ms, 0)}ms</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {loading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                className="min-h-[280px] rounded-[24px] border border-cyan-400/15 bg-cyan-400/[0.04] flex flex-col items-center justify-center gap-4 p-12 text-center"
              >
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-cyan-950 rounded-full animate-spin-slow" />
                  <div className="absolute top-0 w-16 h-16 border-t-4 border-cyan-300 rounded-full animate-spin" />
                  <Cpu className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-cyan-300" />
                </div>
                <div className="space-y-1">
                  <h3 className="text-white font-black text-sm uppercase tracking-[0.24em]">Matching Benchmarks</h3>
                  <p className="text-slate-400 text-xs font-medium">Evaluating complexity, clarity, and safe switching thresholds</p>
                </div>
              </motion.div>
            )}

            {result && <RecommendationOutput data={result} />}

            {error && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-red-950/20 border border-red-500/20 rounded-[24px] px-6 py-5 flex items-center gap-4"
              >
                <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
                <p className="text-red-300 text-sm font-semibold tracking-tight">{error}</p>
              </motion.div>
            )}

            {!loading && !result && !error && (
              <div className="min-h-[280px] rounded-[24px] border-2 border-dashed border-white/10 bg-slate-950/40 p-10 flex flex-col justify-center text-left">
                <p className="text-[10px] font-black uppercase tracking-[0.24em] text-slate-500">Awaiting Analysis</p>
                <h3 className="mt-4 text-2xl font-black text-white">Choose a use case, select a baseline model, and run the recommender.</h3>
                <p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-400">
                  You'll get a cleaner, benchmark-aware comparison with matched slice details, switch rationale, expected metrics, and top alternatives.
                </p>
              </div>
            )}
          </AnimatePresence>

          {optionError && (
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 px-5 py-4">
              <p className="text-sm font-semibold text-amber-200">{optionError}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
