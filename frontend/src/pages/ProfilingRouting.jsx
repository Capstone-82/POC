import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sparkles,
  Zap,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Layers,
  Cpu,
  ShieldAlert,
  ArrowRight,
  Sliders,
  Maximize2,
  Clock,
  DollarSign,
  Award,
  Tag,
  Search,
  BookOpen,
  Code
} from 'lucide-react'
import { profileAndRoute } from '../api/profiling'

const SAMPLE_PROMPTS = [
  {
    label: 'T1 Simple Factual',
    prompt: 'What is horizontal scaling and how does it differ from vertical scaling?',
    tier: 'T1'
  },
  {
    label: 'T2 Medium Analytical',
    prompt: 'Compare PostgreSQL and MongoDB for a real-time e-commerce transaction processing system. Provide a tabular summary of trade-offs, scalability, and ACID compliance.',
    tier: 'T2'
  },
  {
    label: 'T3 Enterprise Strategic',
    prompt: 'Design a multi-cloud GenAI governance architecture for a Fortune 500 financial company, including compliance risks (GDPR, SOC2, EU AI Act), FinOps cost management, and vendor evaluation criteria.',
    tier: 'T3'
  }
]

export default function ProfilingRouting() {
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState('')
  const [includeLegacy, setIncludeLegacy] = useState(false)
  const [topN, setTopN] = useState(3)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const [showPklDetails, setShowPklDetails] = useState(false)
  const [showRejections, setShowRejections] = useState(false)

  const handleRoute = async (promptOverride = null) => {
    const textToRoute = promptOverride !== null ? promptOverride : prompt
    if (!textToRoute.trim()) {
      setError('Please enter a prompt to profile and route.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await profileAndRoute({
        prompt: textToRoute,
        max_tokens: maxTokens || null,
        include_legacy: includeLegacy,
        top_n: topN
      })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect to profiling & routing backend.')
    } finally {
      setLoading(false)
    }
  }

  const getTierGradient = (tier) => {
    switch (tier) {
      case 'T1':
        return 'from-emerald-500/20 via-teal-500/10 to-transparent border-emerald-500/40 text-emerald-400'
      case 'T2':
        return 'from-blue-500/20 via-indigo-500/10 to-transparent border-blue-500/40 text-blue-400'
      case 'T3':
        return 'from-purple-500/20 via-pink-500/10 to-transparent border-purple-500/40 text-purple-400'
      default:
        return 'from-gray-500/20 to-transparent border-gray-500/40 text-gray-400'
    }
  }

  const getTierBadgeClass = (tier) => {
    switch (tier) {
      case 'T1':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
      case 'T2':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'T3':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/30'
      default:
        return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    }
  }

  const dimInfo = [
    { key: 'd1', label: 'Semantic Complexity', desc: 'Syntactic depth, constraint multi-layering', score: result?.prompt_profile?.d1 },
    { key: 'd2', label: 'Domain Specificity', desc: 'Jargon density, vendor/framework terms', score: result?.prompt_profile?.d2 },
    { key: 'd3', label: 'Output Formality', desc: 'Structured deliverable requirements', score: result?.prompt_profile?.d3 },
    { key: 'd4', label: 'Research Dependency', desc: 'Retrieval, live data, external references', score: result?.prompt_profile?.d4 },
    { key: 'd5', label: 'Context Requirement', desc: 'In-prompt document/attachment length', score: result?.prompt_profile?.d5 }
  ]

  return (
    <div className="max-w-7xl mx-auto py-8 space-y-8 pb-16">
      {/* Header Banner */}
      <div className="relative overflow-hidden rounded-2xl glass-card border border-white/10 p-8">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/30 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5" /> XGBoost Multi-Head Router
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                v3 Registry (35 Models)
              </span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight gradient-text">
              Prompt Profiling & Model Routing
            </h1>
            <p className="text-gray-400 text-sm mt-1.5 max-w-2xl">
              Profile raw prompts across 5 complexity dimensions ($D_1$–$D_5$), predict intent, domain & reasoning chain from our trained ML profiler bundle, then route to the optimal lowest-cost LLM.
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Input Panel (Left / Top) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-5">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                <Code className="w-4 h-4 text-blue-400" /> Enter Prompt
              </label>
              <span className="text-xs text-gray-500">
                {prompt.length} chars | {prompt.split(/\s+/).filter(Boolean).length} words
              </span>
            </div>

            <textarea
              rows={6}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Paste your raw prompt here to profile complexity and route to the optimal model..."
              className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/60 transition duration-200 resize-none font-sans"
            />

            {/* Quick Sample Prompts */}
            <div className="space-y-2">
              <span className="text-xs font-medium text-gray-400 uppercase tracking-wider block">
                Quick Sample Prompts:
              </span>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_PROMPTS.map((sample, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setPrompt(sample.prompt)
                      handleRoute(sample.prompt)
                    }}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/15 text-gray-300 hover:text-white border border-white/10 transition flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3 h-3 text-blue-400" />
                    {sample.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Advanced Configuration Controls */}
            <div className="pt-4 border-t border-white/10 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-medium text-gray-400 block mb-1">
                    Max Tokens Override (Optional)
                  </label>
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(e.target.value)}
                    placeholder="Auto (d3 bucket)"
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-gray-600 focus:outline-none focus:border-blue-500/50"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-gray-400 block mb-1">
                    Top Recommendations
                  </label>
                  <select
                    value={topN}
                    onChange={(e) => setTopN(e.target.value)}
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-blue-500/50"
                  >
                    <option value={3}>Top 3 Models</option>
                    <option value={5}>Top 5 Models</option>
                    <option value={10}>Top 10 Models</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center justify-between pt-1">
                <label className="text-xs font-medium text-gray-300 flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeLegacy}
                    onChange={(e) => setIncludeLegacy(e.target.checked)}
                    className="rounded border-white/20 bg-black/40 text-blue-600 focus:ring-0 w-4 h-4 cursor-pointer"
                  />
                  Include Legacy Models in Candidate Pool
                </label>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Action Button */}
            <button
              onClick={() => handleRoute()}
              disabled={loading}
              className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/30 flex items-center justify-center gap-2 transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Profiling & Routing...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Profile Prompt & Route Model</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Panel (Right / Bottom) */}
        <div className="lg:col-span-7 space-y-6">
          {!result && !loading && (
            <div className="glass-card rounded-2xl p-12 border border-white/10 text-center flex flex-col items-center justify-center min-h-[400px]">
              <div className="w-16 h-16 rounded-2xl bg-blue-600/10 flex items-center justify-center mb-4 text-blue-400 border border-blue-500/20">
                <Layers className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">No Profile Executed Yet</h3>
              <p className="text-sm text-gray-400 max-w-md">
                Enter a custom prompt or choose a sample prompt on the left to analyze complexity parameters ($D_1$–$D_5$) and generate cost-effective LLM routing.
              </p>
            </div>
          )}

          {loading && (
            <div className="glass-card rounded-2xl p-12 border border-white/10 text-center flex flex-col items-center justify-center min-h-[400px] space-y-4">
              <div className="w-12 h-12 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin mb-2" />
              <p className="text-white font-semibold text-base">Running XGBoost Profiler & Multi-Stage Filters...</p>
              <p className="text-xs text-gray-400">Computing BGE embeddings, PCA reduction, KNN neighbors, and evaluating 35 model candidates.</p>
            </div>
          )}

          {result && (
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="space-y-6"
              >
                {/* Profile Summary Card */}
                <div className={`glass-card rounded-2xl p-6 border bg-gradient-to-b ${getTierGradient(result.resolved_tier)} space-y-6`}>
                  <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-5">
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wider text-gray-400 block mb-1">
                        Predicted Complexity Tier
                      </span>
                      <div className="flex items-center gap-3">
                        <span className={`px-3.5 py-1.5 rounded-lg text-lg font-black border ${getTierBadgeClass(result.resolved_tier)} shadow-lg`}>
                          {result.resolved_tier} Tier
                        </span>
                        {result.tier_escalated && (
                          <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                            <ShieldAlert className="w-3.5 h-3.5" /> Escalated
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <span className="text-xs font-medium text-gray-400 block">Profiler Confidence</span>
                        <span className="text-lg font-bold text-white">
                          {(result.prompt_profile.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-12 h-12 rounded-full border-4 border-blue-500/30 border-t-blue-400 flex items-center justify-center font-bold text-xs text-blue-300 bg-black/40">
                        {Math.round(result.prompt_profile.confidence * 100)}%
                      </div>
                    </div>
                  </div>

                  {/* Profile Meta Metrics Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-black/30 rounded-xl p-3 border border-white/5">
                      <span className="text-[11px] text-gray-400 block">Domain</span>
                      <span className="text-sm font-semibold text-white truncate block">
                        {result.prompt_profile.domain}
                      </span>
                    </div>

                    <div className="bg-black/30 rounded-xl p-3 border border-white/5">
                      <span className="text-[11px] text-gray-400 block">Intent / Task</span>
                      <span className="text-sm font-semibold text-white truncate block">
                        {result.prompt_profile.intent} / {result.prompt_profile.task_type}
                      </span>
                    </div>

                    <div className="bg-black/30 rounded-xl p-3 border border-white/5">
                      <span className="text-[11px] text-gray-400 block">Reasoning Mode</span>
                      <span className={`text-sm font-semibold block ${result.prompt_profile.reasoning_chain_detected ? 'text-amber-400' : 'text-gray-300'}`}>
                        {result.prompt_profile.reasoning_chain_detected ? 'Required' : 'Standard'}
                      </span>
                    </div>

                    <div className="bg-black/30 rounded-xl p-3 border border-white/5">
                      <span className="text-[11px] text-gray-400 block">Composite Score</span>
                      <span className="text-sm font-semibold text-white block">
                        {result.prompt_profile.complexity_score.toFixed(4)}
                      </span>
                    </div>
                  </div>

                  {/* Escalation Warning banner */}
                  {result.tier_escalated && (
                    <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 flex-shrink-0 text-amber-400" />
                      <span>{result.escalation_reason}</span>
                    </div>
                  )}
                </div>

                {/* Expandable .pkl Profiler Parameters */}
                <div className="glass-card rounded-2xl border border-white/10 overflow-hidden">
                  <button
                    onClick={() => setShowPklDetails(!showPklDetails)}
                    className="w-full p-5 flex items-center justify-between text-left hover:bg-white/5 transition"
                  >
                    <div className="flex items-center gap-2.5">
                      <Sliders className="w-5 h-5 text-blue-400" />
                      <span className="text-sm font-bold text-white">
                        Detailed .pkl Profiler Parameters ($D_1$–$D_5$ Dimensions)
                      </span>
                    </div>
                    {showPklDetails ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                  </button>

                  {showPklDetails && (
                    <div className="p-6 border-t border-white/10 bg-black/20 space-y-6">
                      {/* Dimension Sliders / Progress bars */}
                      <div className="space-y-4">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                          Dimension Scores (Predicted 5-Class Multiclass Probas)
                        </h4>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {dimInfo.map((dim) => (
                            <div key={dim.key} className="bg-black/40 p-3.5 rounded-xl border border-white/5 space-y-2">
                              <div className="flex items-center justify-between text-xs">
                                <span className="font-semibold text-white">
                                  {dim.key.toUpperCase()} - {dim.label}
                                </span>
                                <span className="font-mono font-bold text-blue-400">
                                  {(dim.score || 0).toFixed(2)}
                                </span>
                              </div>

                              <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-400 transition-all duration-500"
                                  style={{ width: `${(dim.score || 0) * 100}%` }}
                                />
                              </div>
                              <span className="text-[11px] text-gray-400 block">{dim.desc}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Formula & Token Estimates */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-white/10">
                        <div className="space-y-1.5">
                          <span className="text-xs font-medium text-gray-400">Composite Score Formula:</span>
                          <p className="text-xs font-mono text-gray-300 bg-black/40 p-2.5 rounded-lg border border-white/5">
                            $D_1 \times 0.35 + D_2 \times 0.20 + D_3 \times 0.20 + D_4 \times 0.15 + D_5 \times 0.10$ = <strong className="text-blue-400">{result.prompt_profile.complexity_score.toFixed(4)}</strong>
                          </p>
                        </div>

                        <div className="space-y-1.5">
                          <span className="text-xs font-medium text-gray-400">Token Volume Estimates:</span>
                          <div className="flex items-center gap-4 text-xs font-mono text-gray-300 bg-black/40 p-2.5 rounded-lg border border-white/5">
                            <span>Input: <strong className="text-white">{result.prompt_profile.input_token_count?.toLocaleString()}</strong> tok</span>
                            <span>Output: <strong className="text-white">{result.prompt_profile.est_output_tokens?.toLocaleString()}</strong> tok</span>
                          </div>
                        </div>
                      </div>

                      {/* Research Signals */}
                      {result.prompt_profile.research_signals?.length > 0 && (
                        <div className="space-y-2 pt-2 border-t border-white/10">
                          <span className="text-xs font-medium text-gray-400 block">Extracted Research Signals:</span>
                          <div className="flex flex-wrap gap-2">
                            {result.prompt_profile.research_signals.map((sig, i) => (
                              <span key={i} className="px-2.5 py-1 rounded-md text-xs font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                                #{sig}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Top Recommended Models Section */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <Award className="w-5 h-5 text-amber-400" />
                      Top {result.recommendations.length} Recommended Models (Filtered & Ranked by Cost)
                    </h3>
                  </div>

                  <div className="space-y-3">
                    {result.recommendations.map((rec) => (
                      <div
                        key={rec.model_id}
                        className={`glass-card rounded-2xl p-5 border transition duration-200 ${
                          rec.rank === 1
                            ? 'border-amber-500/40 bg-gradient-to-r from-amber-500/10 via-transparent to-transparent'
                            : 'border-white/10 hover:border-white/20'
                        }`}
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                          <div className="flex items-center gap-3.5">
                            {/* Rank Badge */}
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-black text-sm ${
                              rec.rank === 1
                                ? 'bg-amber-500 text-black shadow-lg shadow-amber-500/30'
                                : rec.rank === 2
                                ? 'bg-gray-300 text-black'
                                : 'bg-amber-700/60 text-white'
                            }`}>
                              #{rec.rank}
                            </div>

                            <div>
                              <div className="flex items-center gap-2">
                                <h4 className="text-base font-bold text-white">{rec.model_id}</h4>
                                <span className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${getTierBadgeClass(rec.tier)}`}>
                                  {rec.tier}
                                </span>
                              </div>
                              <span className="text-xs text-gray-400 capitalize">Provider: {rec.provider}</span>
                            </div>
                          </div>

                          {/* Cost Metric */}
                          <div className="text-left sm:text-right">
                            <span className="text-xs text-gray-400 block">Est. Cost / Request</span>
                            <span className="text-base font-extrabold text-emerald-400 font-mono">
                              ${rec.estimated_cost_usd.toFixed(6)}
                            </span>
                          </div>
                        </div>

                        {/* Reasons & Strengths */}
                        {rec.reasons?.length > 0 && (
                          <div className="mt-4 pt-3 border-t border-white/10 flex flex-wrap gap-2">
                            {rec.reasons.map((reason, idx) => (
                              <span key={idx} className="px-2.5 py-1 rounded-md text-xs bg-white/5 text-gray-300 border border-white/10 flex items-center gap-1.5">
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                {reason}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Expandable Rejections Audit Trail */}
                <div className="glass-card rounded-2xl border border-white/10 overflow-hidden">
                  <button
                    onClick={() => setShowRejections(!showRejections)}
                    className="w-full p-5 flex items-center justify-between text-left hover:bg-white/5 transition"
                  >
                    <div className="flex items-center gap-2.5 text-gray-300">
                      <XCircle className="w-5 h-5 text-red-400" />
                      <span className="text-sm font-bold text-white">
                        View Rejection Audit Trail ({Object.keys(result.rejections || {}).length} Models Excluded)
                      </span>
                    </div>
                    {showRejections ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                  </button>

                  {showRejections && (
                    <div className="p-6 border-t border-white/10 bg-black/30 space-y-2 max-h-96 overflow-y-auto font-mono text-xs">
                      {Object.entries(result.rejections || {}).map(([modelId, reason]) => (
                        <div key={modelId} className="flex items-start gap-3 py-1.5 border-b border-white/5 last:border-0">
                          <span className="text-red-400 font-bold flex-shrink-0 w-40 truncate">{modelId}</span>
                          <span className="text-gray-400">{reason}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            </AnimatePresence>
          )}
        </div>
      </div>
    </div>
  )
}
