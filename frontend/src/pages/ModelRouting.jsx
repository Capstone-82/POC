import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield, Plus, Trash2, Save, FlaskConical, ChevronDown,
  ChevronUp, CheckCircle2, XCircle, AlertCircle, Loader2,
  Settings, List, SlidersHorizontal, Zap, RefreshCw, Edit3, X
} from 'lucide-react'
import {
  fetchApps, upsertApp, deleteApp, testRoute, fetchAvailableModels
} from '../api/modelRoutingApi'

/* ─── Tiny helpers ───────────────────────────────────────────── */
const ENVS       = ['dev', 'prod', 'staging']
const BADGE_ENV  = { prod: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', dev: 'bg-amber-500/20 text-amber-400 border-amber-500/30', staging: 'bg-purple-500/20 text-purple-400 border-purple-500/30' }
const clx        = (...c) => c.filter(Boolean).join(' ')

function Badge({ label, env }) {
  return (
    <span className={clx('text-xs px-2 py-0.5 rounded-full border font-semibold', BADGE_ENV[env] ?? 'bg-white/10 text-gray-400 border-white/10')}>
      {label}
    </span>
  )
}

function Spinner() {
  return <Loader2 className="w-4 h-4 animate-spin" />
}

function SectionTitle({ icon: Icon, title, subtitle }) {
  return (
    <div className="flex items-start gap-3 mb-6">
      <div className="p-2 rounded-xl bg-blue-600/20 border border-blue-500/30 mt-0.5">
        <Icon className="w-5 h-5 text-blue-400" />
      </div>
      <div>
        <h2 className="text-lg font-bold text-white">{title}</h2>
        {subtitle && <p className="text-sm text-gray-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  )
}

/* ─── Empty-state app config template ───────────────────────── */
const BLANK_APP = {
  app_id: '',
  env: 'prod',
  allowed_models: [],
  context_limits: { max_input_tokens: 8000, max_output_tokens: 4096, max_total_tokens: 10000 },
  throttle: { rate_limit: 100, burst_limit: 200, quota_per_day: 10000 },
}

/* ══════════════════════════════════════════════════════════════
   SECTION 1 — App Config Table
══════════════════════════════════════════════════════════════ */
function AppConfigTable({ apps, availableModels, onSave, onDelete, loading }) {
  const [expanded, setExpanded] = useState(null)  // which app_id/env is open
  const [editing,  setEditing]  = useState(null)  // draft being edited
  const [adding,   setAdding]   = useState(false)
  const [draft,    setDraft]    = useState(BLANK_APP)
  const [saving,   setSaving]   = useState(false)
  const [deleting, setDeleting] = useState(null)

  const key = (a) => `${a.app_id}/${a.env}`

  const startEdit = (app) => {
    setEditing(key(app))
    setDraft(JSON.parse(JSON.stringify(app)))
    setAdding(false)
  }

  const cancelEdit = () => { setEditing(null); setAdding(false); setDraft(BLANK_APP) }

  const handleSave = async () => {
    setSaving(true)
    try { await onSave(draft); cancelEdit() }
    finally { setSaving(false) }
  }

  const handleDelete = async (app) => {
    setDeleting(key(app))
    try { await onDelete(app.app_id, app.env) }
    finally { setDeleting(null) }
  }

  const toggleModel = (modelId) => {
    setDraft(d => ({
      ...d,
      allowed_models: d.allowed_models.includes(modelId)
        ? d.allowed_models.filter(m => m !== modelId)
        : [...d.allowed_models, modelId],
    }))
  }

  const EditForm = ({ isNew }) => (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className="rounded-xl border border-blue-500/30 bg-blue-600/5 p-5 mb-4"
    >
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-gray-400 mb-1 block">App ID</label>
          <input
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-blue-500/60 transition"
            value={draft.app_id}
            onChange={e => setDraft(d => ({ ...d, app_id: e.target.value }))}
            placeholder="e.g. modelmatrix"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 mb-1 block">Environment</label>
          <select
            className="w-full bg-[#0f172a] border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-blue-500/60 transition"
            value={draft.env}
            onChange={e => setDraft(d => ({ ...d, env: e.target.value }))}
          >
            {ENVS.map(e => <option key={e} value={e}>{e}</option>)}
          </select>
        </div>
      </div>

      {/* Context Limits */}
      <p className="text-xs text-gray-400 font-semibold uppercase tracking-widest mb-2">Context Limits</p>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {['max_input_tokens','max_output_tokens','max_total_tokens'].map(k => (
          <div key={k}>
            <label className="text-xs text-gray-500 mb-1 block">{k.replace(/_/g,' ')}</label>
            <input
              type="number" min={1}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-blue-500/60 transition"
              value={draft.context_limits[k]}
              onChange={e => setDraft(d => ({ ...d, context_limits: { ...d.context_limits, [k]: Number(e.target.value) } }))}
            />
          </div>
        ))}
      </div>

      {/* Throttle */}
      <p className="text-xs text-gray-400 font-semibold uppercase tracking-widest mb-2">Throttle</p>
      <div className="grid grid-cols-3 gap-3 mb-5">
        {['rate_limit','burst_limit','quota_per_day'].map(k => (
          <div key={k}>
            <label className="text-xs text-gray-500 mb-1 block">{k.replace(/_/g,' ')}</label>
            <input
              type="number" min={1}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none focus:border-blue-500/60 transition"
              value={draft.throttle[k]}
              onChange={e => setDraft(d => ({ ...d, throttle: { ...d.throttle, [k]: Number(e.target.value) } }))}
            />
          </div>
        ))}
      </div>

      {/* Allow-list */}
      <p className="text-xs text-gray-400 font-semibold uppercase tracking-widest mb-2">Allow-List</p>
      <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto pr-1 mb-5">
        {availableModels.map(m => {
          const on = draft.allowed_models.includes(m)
          return (
            <button
              key={m}
              onClick={() => toggleModel(m)}
              className={clx(
                'text-xs px-3 py-1.5 rounded-lg border font-mono transition-all',
                on
                  ? 'bg-blue-600/30 border-blue-500/50 text-blue-300'
                  : 'bg-white/5 border-white/10 text-gray-500 hover:text-gray-300 hover:border-white/20'
              )}
            >
              {on && <span className="mr-1">✓</span>}{m}
            </button>
          )
        })}
      </div>

      <div className="flex gap-2 justify-end">
        <button onClick={cancelEdit} className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 transition flex items-center gap-1.5">
          <X className="w-3.5 h-3.5" /> Cancel
        </button>
        <button
          onClick={handleSave} disabled={saving || !draft.app_id}
          className="px-4 py-2 rounded-lg text-sm bg-blue-600 hover:bg-blue-500 text-white font-semibold transition flex items-center gap-1.5 disabled:opacity-50"
        >
          {saving ? <Spinner /> : <Save className="w-3.5 h-3.5" />} Save
        </button>
      </div>
    </motion.div>
  )

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <SectionTitle icon={List} title="App Configurations" subtitle="Manage per-app allow-lists and limits" />
        <button
          onClick={() => { setAdding(true); setEditing(null); setDraft(BLANK_APP) }}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition shadow-lg shadow-blue-900/30"
        >
          <Plus className="w-4 h-4" /> New App
        </button>
      </div>

      <AnimatePresence>
        {adding && <EditForm isNew />}
      </AnimatePresence>

      {loading ? (
        <div className="flex items-center justify-center h-32 text-gray-500 gap-2">
          <Spinner /> Loading configs…
        </div>
      ) : apps.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/10 p-10 text-center text-gray-500">
          No app configurations yet. Click <strong className="text-gray-300">New App</strong> to add one.
        </div>
      ) : (
        <div className="space-y-3">
          {apps.map(app => {
            const k = key(app)
            const isExpanded = expanded === k
            const isEditing  = editing === k
            return (
              <motion.div
                key={k}
                layout
                className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden"
              >
                {/* Row header */}
                <div
                  className="flex items-center justify-between px-5 py-4 cursor-pointer hover:bg-white/[0.03] transition"
                  onClick={() => setExpanded(isExpanded ? null : k)}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-white/5"><Shield className="w-4 h-4 text-blue-400" /></div>
                    <div>
                      <p className="font-semibold text-white">{app.app_id}</p>
                      <p className="text-xs text-gray-500">{app.allowed_models?.length ?? 0} models allowed</p>
                    </div>
                    <Badge label={app.env} env={app.env} />
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={e => { e.stopPropagation(); startEdit(app) }}
                      className="p-1.5 rounded-lg hover:bg-white/10 text-gray-500 hover:text-blue-400 transition"
                    >
                      <Edit3 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); handleDelete(app) }}
                      disabled={deleting === k}
                      className="p-1.5 rounded-lg hover:bg-red-500/10 text-gray-500 hover:text-red-400 transition"
                    >
                      {deleting === k ? <Spinner /> : <Trash2 className="w-4 h-4" />}
                    </button>
                    {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
                  </div>
                </div>

                {/* Expanded detail */}
                <AnimatePresence>
                  {isExpanded && !isEditing && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25 }}
                      className="overflow-hidden border-t border-white/10"
                    >
                      <div className="px-5 py-4 grid grid-cols-3 gap-6 text-sm">
                        <div>
                          <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Context Limits</p>
                          {Object.entries(app.context_limits ?? {}).map(([k, v]) => (
                            <div key={k} className="flex justify-between text-gray-300 mb-1">
                              <span className="text-gray-500">{k.replace(/_/g,' ')}</span>
                              <span className="font-mono">{v?.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                        <div>
                          <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Throttle</p>
                          {Object.entries(app.throttle ?? {}).map(([k, v]) => (
                            <div key={k} className="flex justify-between text-gray-300 mb-1">
                              <span className="text-gray-500">{k.replace(/_/g,' ')}</span>
                              <span className="font-mono">{v?.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                        <div>
                          <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Allowed Models</p>
                          <div className="flex flex-wrap gap-1.5">
                            {(app.allowed_models ?? []).map(m => (
                              <span key={m} className="text-xs bg-blue-600/15 text-blue-300 border border-blue-500/25 rounded px-2 py-0.5 font-mono">{m}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                  {isEditing && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="border-t border-white/10 p-5"
                    >
                      <EditForm />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   SECTION 2 — Gateway Tester
══════════════════════════════════════════════════════════════ */
function GatewayTester({ apps, availableModels }) {
  const [appId,     setAppId]     = useState('')
  const [env,       setEnv]       = useState('prod')
  const [modelId,   setModelId]   = useState('')
  const [prompt,    setPrompt]    = useState('')
  const [maxOut,    setMaxOut]    = useState(512)
  const [result,    setResult]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)

  const appOptions = [...new Set(apps.map(a => a.app_id))]

  const run = async () => {
    if (!appId || !modelId || !prompt) return
    setLoading(true); setResult(null); setError(null)
    try {
      const res = await testRoute({ app_id: appId, env, model_id: modelId, prompt, max_output_tokens: maxOut })
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const DecisionBadge = ({ decision }) => {
    const map = {
      allowed:  { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/15 border-emerald-500/30', label: 'Allowed' },
      rejected: { icon: XCircle,      color: 'text-red-400',     bg: 'bg-red-500/15 border-red-500/30',         label: 'Rejected' },
      throttled:{ icon: AlertCircle,  color: 'text-amber-400',   bg: 'bg-amber-500/15 border-amber-500/30',     label: 'Throttled' },
    }
    const d = map[decision] ?? map.rejected
    return (
      <div className={clx('inline-flex items-center gap-2 px-4 py-2 rounded-full border font-bold text-sm', d.bg, d.color)}>
        <d.icon className="w-4 h-4" /> {d.label}
      </div>
    )
  }

  return (
    <div>
      <SectionTitle icon={FlaskConical} title="Gateway Tester" subtitle="Dry-run a routing decision without invoking any model" />
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs text-gray-400 mb-1.5 block">Application</label>
          <select
            className="w-full bg-[#0f172a] border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white outline-none focus:border-blue-500/60 transition"
            value={appId} onChange={e => setAppId(e.target.value)}
          >
            <option value="">Select app…</option>
            {appOptions.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 mb-1.5 block">Environment</label>
          <select
            className="w-full bg-[#0f172a] border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white outline-none focus:border-blue-500/60 transition"
            value={env} onChange={e => setEnv(e.target.value)}
          >
            {ENVS.map(e => <option key={e} value={e}>{e}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 mb-1.5 block">Model</label>
          <select
            className="w-full bg-[#0f172a] border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white outline-none focus:border-blue-500/60 transition"
            value={modelId} onChange={e => setModelId(e.target.value)}
          >
            <option value="">Select model…</option>
            {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 mb-1.5 block">Max Output Tokens</label>
          <input
            type="number" min={1}
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white outline-none focus:border-blue-500/60 transition"
            value={maxOut} onChange={e => setMaxOut(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="mb-4">
        <label className="text-xs text-gray-400 mb-1.5 block">Prompt</label>
        <textarea
          rows={4}
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white outline-none focus:border-blue-500/60 transition resize-none font-mono"
          placeholder="Enter a prompt to test routing…"
          value={prompt} onChange={e => setPrompt(e.target.value)}
        />
      </div>

      <button
        onClick={run}
        disabled={loading || !appId || !modelId || !prompt}
        className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-sm transition-all shadow-lg shadow-blue-900/30 flex items-center justify-center gap-2 disabled:opacity-40"
      >
        {loading ? <><Spinner /> Testing…</> : <><Zap className="w-4 h-4" /> Test Route</>}
      </button>

      {/* Result */}
      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {error}
          </motion.div>
        )}
        {result && (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mt-5 rounded-2xl border border-white/10 bg-white/[0.03] p-6 space-y-5"
          >
            <div className="flex items-center justify-between">
              <DecisionBadge decision={result.decision} />
              <span className="text-xs text-gray-500 font-mono">{result.model_id}</span>
            </div>

            <p className={clx('text-sm font-medium', result.decision === 'allowed' ? 'text-emerald-300' : 'text-red-300')}>
              {result.message}
            </p>

            {/* Token breakdown */}
            <div className="grid grid-cols-3 gap-3">
              {[
                ['Input', result.estimated_input_tokens, result.limits?.max_input_tokens],
                ['Output', result.estimated_output_tokens, result.limits?.max_output_tokens],
                ['Total', result.estimated_total_tokens, result.limits?.max_total_tokens],
              ].map(([label, val, max]) => {
                const pct = max ? Math.min(100, Math.round((val / max) * 100)) : 0
                const over = val > max
                return (
                  <div key={label} className="rounded-xl bg-white/5 border border-white/10 p-4">
                    <p className="text-xs text-gray-500 mb-1">{label} Tokens</p>
                    <p className={clx('text-xl font-bold font-mono', over ? 'text-red-400' : 'text-white')}>
                      {val?.toLocaleString()}
                    </p>
                    <p className="text-xs text-gray-600">limit {max?.toLocaleString()}</p>
                    <div className="mt-2 h-1.5 rounded-full bg-white/10 overflow-hidden">
                      <div
                        className={clx('h-full rounded-full transition-all', over ? 'bg-red-500' : 'bg-blue-500')}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Throttle info (only if allowed) */}
            {result.decision === 'allowed' && result.throttle && (
              <div className="rounded-xl bg-white/5 border border-white/10 p-4">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Throttle Settings</p>
                <div className="grid grid-cols-3 gap-3 text-sm text-center">
                  <div><p className="text-xs text-gray-500">Rate</p><p className="font-bold text-white">{result.throttle.rate_limit} <span className="text-gray-500 font-normal text-xs">req/s</span></p></div>
                  <div><p className="text-xs text-gray-500">Burst</p><p className="font-bold text-white">{result.throttle.burst_limit} <span className="text-gray-500 font-normal text-xs">req/s</span></p></div>
                  <div><p className="text-xs text-gray-500">Daily Quota</p><p className="font-bold text-white">{result.throttle.quota_per_day?.toLocaleString()}</p></div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   MAIN PAGE
══════════════════════════════════════════════════════════════ */
export default function ModelRouting() {
  const [apps,            setApps]           = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [appsLoading,     setAppsLoading]     = useState(true)
  const [toast,           setToast]           = useState(null)

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  const loadApps = useCallback(async () => {
    setAppsLoading(true)
    try { setApps(await fetchApps()) }
    catch { showToast('Failed to load app configs', 'error') }
    finally { setAppsLoading(false) }
  }, [])

  useEffect(() => {
    loadApps()
    fetchAvailableModels().then(setAvailableModels).catch(() => {})
  }, [loadApps])

  const handleSave = async (config) => {
    await upsertApp(config)
    await loadApps()
    showToast(`Saved ${config.app_id}/${config.env}`)
  }

  const handleDelete = async (appId, env) => {
    await deleteApp(appId, env)
    await loadApps()
    showToast(`Deleted ${appId}/${env}`)
  }

  return (
    <div className="max-w-6xl mx-auto py-10 px-4 space-y-8">

      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <div className="flex items-center gap-4 mb-2">
          <div className="p-3 rounded-2xl bg-gradient-to-br from-blue-600/30 to-indigo-600/20 border border-blue-500/30 shadow-lg shadow-blue-900/20">
            <Shield className="w-7 h-7 text-blue-400" />
          </div>
          <div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">AWS Model Routing</h1>
            <p className="text-gray-400 text-sm mt-0.5">
              API Gateway · Lambda Allow-List · Context Window Limits · Throttling
            </p>
          </div>
          <button onClick={loadApps} className="ml-auto p-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition" title="Refresh">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Architecture strip */}
        <div className="mt-5 flex items-center gap-0 overflow-x-auto pb-1">
          {[
            { label: 'Application', sub: 'Any client' },
            { label: 'API Gateway', sub: 'Throttling' },
            { label: 'Lambda', sub: 'Allow-List + Context' },
            { label: 'Bedrock / Vertex', sub: 'Model invocation' },
          ].map((step, i, arr) => (
            <div key={step.label} className="flex items-center shrink-0">
              <div className="px-4 py-2.5 rounded-xl bg-white/[0.04] border border-white/10 text-center min-w-[120px]">
                <p className="text-xs font-bold text-white">{step.label}</p>
                <p className="text-[10px] text-gray-500 mt-0.5">{step.sub}</p>
              </div>
              {i < arr.length - 1 && (
                <div className="flex items-center px-1.5">
                  <div className="h-px w-6 bg-white/20" />
                  <div className="w-0 h-0 border-t-4 border-b-4 border-l-4 border-transparent border-l-white/20" />
                </div>
              )}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Left — App configs */}
        <motion.div
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
          className="rounded-2xl border border-white/10 bg-white/[0.03] p-6"
        >
          <AppConfigTable
            apps={apps}
            availableModels={availableModels}
            onSave={handleSave}
            onDelete={handleDelete}
            loading={appsLoading}
          />
        </motion.div>

        {/* Right — Tester */}
        <motion.div
          initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.2 }}
          className="rounded-2xl border border-white/10 bg-white/[0.03] p-6"
        >
          <GatewayTester apps={apps} availableModels={availableModels} />
        </motion.div>
      </div>

      {/* Toast */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }}
            className={clx(
              'fixed bottom-6 right-6 px-5 py-3 rounded-xl border text-sm font-semibold shadow-2xl flex items-center gap-2 z-50',
              toast.type === 'error'
                ? 'bg-red-500/20 border-red-500/40 text-red-300'
                : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
            )}
          >
            {toast.type === 'error' ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
            {toast.msg}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
