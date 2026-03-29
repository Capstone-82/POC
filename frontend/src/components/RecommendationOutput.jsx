import {
  Sparkles,
  Cpu,
  Layers,
  ShieldCheck,
  Database,
  ArrowRight,
  AlertTriangle,
} from 'lucide-react'
import { motion } from 'framer-motion'

const fmtNum = (value, digits = 2) => {
  if (typeof value !== 'number') return 'N/A'
  return value.toFixed(digits)
}

const fmtPct = (value) => {
  if (typeof value !== 'number') return 'N/A'
  return `${value > 0 ? '+' : ''}${value}%`
}

function Metric({ label, value }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-3 min-w-0">
      <p className="text-[10px] font-black uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-1 text-base font-black text-white truncate">{value}</p>
    </div>
  )
}

function Delta({ label, value, positiveGood = false }) {
  const good =
    typeof value === 'number'
      ? positiveGood
        ? value >= 0
        : value <= 0
      : null

  const tone =
    good === null
      ? 'border-white/10 bg-white/[0.03] text-slate-300'
      : good
        ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200'
        : 'border-rose-400/20 bg-rose-400/10 text-rose-200'

  return (
    <div className={`rounded-xl border px-3 py-3 ${tone}`}>
      <p className="text-[10px] font-black uppercase tracking-[0.14em] opacity-70">{label}</p>
      <p className="mt-1 text-base font-black">{fmtPct(value)}</p>
    </div>
  )
}

export default function RecommendationOutput({ data }) {
  const currentStats = data.current_model_stats
  const warnings = data.warnings || []
  const sourceLabel = data.data_source === 'supabase' ? 'Supabase' : 'Local CSV'

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.95),rgba(10,16,28,0.98))] shadow-[0_24px_100px_rgba(2,8,23,0.45)] overflow-hidden"
    >
      <div className="border-b border-white/8 px-6 py-5 lg:px-7">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-cyan-300" />
              <h3 className="text-2xl font-black text-white tracking-tight">Recommendation Analysis</h3>
            </div>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Using the <span className="font-bold text-white">{data.recommendation_mode}</span> policy with matched benchmark slices and safe switching thresholds.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <div className="rounded-full border border-yellow-400/20 bg-yellow-400/10 px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] text-yellow-200 flex items-center gap-2">
              <Layers className="w-3.5 h-3.5" />
              {data.complexity}
            </div>
            <div className="rounded-full border border-blue-400/20 bg-blue-400/10 px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] text-blue-200 flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5" />
              {data.clarity}
            </div>
            <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] text-emerald-200 flex items-center gap-2">
              <ShieldCheck className="w-3.5 h-3.5" />
              {data.filter_level}
            </div>
            <div className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] text-slate-300 flex items-center gap-2">
              <Database className="w-3.5 h-3.5" />
              {sourceLabel}
            </div>
          </div>
        </div>
      </div>

      <div className="px-6 py-6 lg:px-7 space-y-5">
        <div className="rounded-[24px] border border-cyan-400/15 bg-cyan-400/[0.05] p-5">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <p className="text-[10px] font-black uppercase tracking-[0.16em] text-cyan-300">Recommended</p>
              <h4 className="mt-2 text-3xl font-black text-white leading-[1.05] break-words">
                {data.recommended_provider}/{data.recommended_model}
              </h4>
            </div>
            <div className="rounded-full bg-cyan-400/20 px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.16em] text-cyan-100">
              {data.switch_recommended ? 'Switch' : 'Stay'}
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Metric label="Accuracy" value={fmtNum(data.expected_accuracy)} />
            <Metric label="Cost" value={`$${fmtNum(data.expected_cost, 6)}`} />
            <Metric label="Latency" value={`${fmtNum(data.expected_latency, 0)}ms`} />
          </div>

          <div className="mt-4 flex flex-wrap gap-2 text-sm text-slate-300">
            <div className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2">
              {data.sample_size} rows for this model
            </div>
            <div className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2">
              {data.slice_row_count} rows in matched slice
            </div>
            <div className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2">
              {data.models_considered} supported models
            </div>
          </div>
        </div>

        <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5">
          <p className="text-[10px] font-black uppercase tracking-[0.16em] text-slate-500">Compared Against</p>
          <div className="mt-3 flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <h4 className="text-2xl font-black text-white break-words">{data.current_model}</h4>
              <p className="mt-1 text-sm text-slate-400">
                {data.current_model_found ? 'Matched in current slice.' : 'No direct match found in current slice.'}
              </p>
            </div>
            <ArrowRight className="w-5 h-5 text-slate-600 shrink-0" />
          </div>

          {currentStats && (
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <Metric label="Accuracy" value={fmtNum(currentStats.avg_accuracy)} />
              <Metric label="Cost" value={`$${fmtNum(currentStats.median_cost, 6)}`} />
              <Metric label="Latency" value={`${fmtNum(currentStats.median_latency_ms, 0)}ms`} />
            </div>
          )}

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <Delta label="Accuracy" value={data.accuracy_delta_pct} positiveGood />
            <Delta label="Cost" value={data.cost_delta_pct} />
            <Delta label="Latency" value={data.latency_delta_pct} />
          </div>
        </div>

        <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5">
          <p className="text-[10px] font-black uppercase tracking-[0.16em] text-slate-500">Rationale</p>
          <p className="mt-3 text-lg leading-relaxed text-slate-100 font-semibold italic">
            "{data.reason}"
          </p>

          <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-4">
            <p className="text-[10px] font-black uppercase tracking-[0.16em] text-slate-500">Policy Summary</p>
            <p className="mt-2 text-sm text-slate-300">{data.policy_reason}</p>
          </div>
        </div>

        {warnings.length > 0 && (
          <div className="rounded-2xl border border-amber-400/20 bg-amber-400/[0.06] px-4 py-4 space-y-2">
            {warnings.map((warning) => (
              <div key={warning} className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-300 shrink-0 mt-0.5" />
                <p className="text-sm text-amber-100">{warning}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}
