import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Coins,
  GitCompareArrows,
  Layers3,
  Terminal,
  Timer,
  TriangleAlert,
  Workflow,
} from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'

const STEP_ORDER = [
  'hash_computed',
  'prompt_logged',
  'embedding_cached',
  'models_selected',
  'inference_started',
  'benchmark_saved',
]

const STEP_LABELS = {
  hash_computed: 'Hash computed',
  prompt_logged: 'Prompt logged',
  embedding_cached: 'Embedding cached',
  models_selected: 'Models selected',
  inference_started: 'Inference started',
  benchmark_saved: 'Benchmark saved',
}

function statusTone(done) {
  return done
    ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
    : 'border-white/10 bg-white/5 text-gray-400'
}

function formatPromptLabel(log) {
  if (typeof log.prompt_index === 'number' && typeof log.total === 'number') {
    return `Prompt ${String(log.prompt_index).padStart(2, '0')} of ${log.total}`
  }
  return 'Prompt'
}

function GlobalStageCard({ log }) {
  const isDone = log.type === 'postprocess_done'
  const label = isDone ? 'Completed' : 'Running'

  return (
    <div className="rounded-2xl border border-cyan-500/10 bg-cyan-500/5 p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Terminal className="h-4 w-4 text-cyan-300" />
          <div>
            <div className="text-xs font-black uppercase tracking-[0.22em] text-cyan-300">
              Global Pipeline
            </div>
            <div className="mt-1 text-sm font-semibold text-white">
              {label} {String(log.stage || '').replaceAll('_', ' ')}
            </div>
          </div>
        </div>
        <span className="rounded-lg border border-cyan-500/20 bg-cyan-500/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-cyan-200">
          {label}
        </span>
      </div>
      <div className="mt-3 text-[11px] text-cyan-100/70">
        {log.stage === 'pairwise' && (
          <span>
            Prompt groups: {log.prompt_groups ?? log.prompt_count ?? 0}. Pairs completed:{' '}
            {log.pairs_completed ?? 0}/{log.pairs_total ?? 0}
          </span>
        )}
        {log.stage === 'win_rates' && <span>Rows written: {log.rows_written ?? 0}</span>}
      </div>
    </div>
  )
}

function StepTimeline({ steps, selectedModels, progressCount, pairwiseCount, failedCount }) {
  return (
    <div className="space-y-3">
      <div className="text-[10px] font-black uppercase tracking-[0.22em] text-gray-500">
        Pipeline Steps
      </div>
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {STEP_ORDER.map((stepKey) => {
          const done = steps.has(stepKey)
          const Icon = done ? CheckCircle2 : Circle
          return (
            <div
              key={stepKey}
              className={`rounded-xl border px-3 py-2 ${statusTone(done)}`}
            >
              <div className="flex items-center gap-2">
                <Icon className="h-3.5 w-3.5" />
                <span className="text-xs font-semibold">{STEP_LABELS[stepKey]}</span>
              </div>
            </div>
          )
        })}
        <div className={`rounded-xl border px-3 py-2 ${statusTone(selectedModels.length > 0)}`}>
          <div className="flex items-center gap-2">
            <Layers3 className="h-3.5 w-3.5" />
            <span className="text-xs font-semibold">{selectedModels.length} models selected</span>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 ${statusTone(progressCount > 0)}`}>
          <div className="flex items-center gap-2">
            <Workflow className="h-3.5 w-3.5" />
            <span className="text-xs font-semibold">{progressCount} responses stored</span>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 ${statusTone(pairwiseCount > 0)}`}>
          <div className="flex items-center gap-2">
            <GitCompareArrows className="h-3.5 w-3.5" />
            <span className="text-xs font-semibold">{pairwiseCount} pairwise decisions</span>
          </div>
        </div>
        <div className={`rounded-xl border px-3 py-2 ${statusTone(failedCount === 0 && steps.has('inference_started'))}`}>
          <div className="flex items-center gap-2">
            <TriangleAlert className="h-3.5 w-3.5" />
            <span className="text-xs font-semibold">
              {failedCount === 0 ? 'No model failures' : `${failedCount} model failures`}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

function ModelChip({ modelId }) {
  return (
    <span className="rounded-lg border border-violet-500/20 bg-violet-500/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-violet-200">
      {modelId}
    </span>
  )
}

function PromptCard({ group, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const { meta, steps, selectedModels, progressLogs, failedLogs, pairwiseResults } = group
  const allCompared = selectedModels.length >= 2 && pairwiseResults.length === (selectedModels.length * (selectedModels.length - 1)) / 2

  return (
    <div className="rounded-3xl border border-white/10 bg-[#07111f]/80 backdrop-blur-3xl shadow-2xl shadow-blue-950/20">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-start justify-between gap-4 rounded-3xl px-5 py-5 text-left"
      >
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-lg border border-blue-500/20 bg-blue-500/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-blue-200">
              {formatPromptLabel(meta)}
            </span>
            <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-gray-300">
              {meta.use_case}
            </span>
            <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-gray-300">
              {meta.prompt_complexity}
            </span>
            <span className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-gray-300">
              {meta.clarity}
            </span>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
            <div className="font-semibold text-white">
              {selectedModels.length} models, {progressLogs.length} stored responses, {pairwiseResults.length} pairwise results
            </div>
            <span
              className={`rounded-lg px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] ${
                allCompared
                  ? 'border border-emerald-500/20 bg-emerald-500/10 text-emerald-200'
                  : 'border border-amber-500/20 bg-amber-500/10 text-amber-200'
              }`}
            >
              {allCompared ? 'Pairwise complete' : 'In progress'}
            </span>
          </div>

          <div className="mt-3 text-[11px] uppercase tracking-[0.18em] text-gray-500">
            {meta.prompt_hash}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden text-right sm:block">
            <div className="text-[10px] font-black uppercase tracking-[0.2em] text-gray-500">
              Expand
            </div>
            <div className="mt-1 text-sm font-semibold text-gray-300">
              {open ? 'Hide details' : 'Show details'}
            </div>
          </div>
          {open ? <ChevronDown className="h-5 w-5 text-gray-300" /> : <ChevronRight className="h-5 w-5 text-gray-300" />}
        </div>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-t border-white/5"
          >
            <div className="space-y-6 px-5 py-5">
              <StepTimeline
                steps={steps}
                selectedModels={selectedModels}
                progressCount={progressLogs.length}
                pairwiseCount={pairwiseResults.length}
                failedCount={failedLogs.length}
              />

              <div className="space-y-3">
                <div className="text-[10px] font-black uppercase tracking-[0.22em] text-gray-500">
                  Selected Models
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedModels.map((modelId) => (
                    <ModelChip key={modelId} modelId={modelId} />
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-[10px] font-black uppercase tracking-[0.22em] text-gray-500">
                  Model Runs
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  {progressLogs.map((log, index) => (
                    <div key={`${log.model_id}-${index}`} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-bold text-white">{log.model_id}</div>
                          <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-gray-500">
                            {log.provider}
                          </div>
                        </div>
                        <span className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-emerald-200">
                          Stored
                        </span>
                      </div>
                      <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
                        <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                          <div className="flex items-center gap-2 text-gray-400">
                            <Coins className="h-3.5 w-3.5 text-emerald-300" />
                            Cost
                          </div>
                          <div className="mt-2 font-black text-white">${Number(log.cost || 0).toFixed(4)}</div>
                        </div>
                        <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                          <div className="flex items-center gap-2 text-gray-400">
                            <Timer className="h-3.5 w-3.5 text-amber-300" />
                            Latency
                          </div>
                          <div className="mt-2 font-black text-white">{log.latency_ms}ms</div>
                        </div>
                        <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                          <div className="flex items-center gap-2 text-gray-400">
                            <Terminal className="h-3.5 w-3.5 text-blue-300" />
                            Tokens
                          </div>
                          <div className="mt-2 font-black text-white">{log.tokens}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {failedLogs.length > 0 && (
                <div className="space-y-3">
                  <div className="text-[10px] font-black uppercase tracking-[0.22em] text-gray-500">
                    Failed Models
                  </div>
                  <div className="grid gap-3 lg:grid-cols-2">
                    {failedLogs.map((log, index) => (
                      <div key={`${log.model_id}-${index}`} className="rounded-2xl border border-orange-500/20 bg-orange-500/10 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-bold text-white">{log.model_id}</div>
                            <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-orange-200/70">
                              {log.provider}
                            </div>
                          </div>
                          <TriangleAlert className="h-4 w-4 text-orange-300" />
                        </div>
                        <div className="mt-3 text-xs text-orange-100/80">{log.reason || 'Model run failed'}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="space-y-3">
                <div className="text-[10px] font-black uppercase tracking-[0.22em] text-gray-500">
                  Pairwise Decisions
                </div>
                <div className="space-y-3">
                  {pairwiseResults.map((log, index) => {
                    const winnerLabel = log.winner_model === 'TIE' ? 'Tie' : `${log.winner_model} won`
                    return (
                      <div key={`${log.model_a}-${log.model_b}-${index}`} className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-4">
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                          <div className="flex items-center gap-3">
                            <GitCompareArrows className="h-4 w-4 text-amber-200" />
                            <div className="text-sm font-semibold text-white">
                              {log.model_a} vs {log.model_b}
                            </div>
                          </div>
                          <span className="w-fit rounded-lg border border-amber-400/20 bg-black/20 px-2 py-1 text-[10px] font-black uppercase tracking-[0.18em] text-amber-100">
                            {winnerLabel}
                          </span>
                        </div>
                        <div className="mt-3 text-[11px] uppercase tracking-[0.16em] text-amber-100/65">
                          judge model: {log.judge_model}
                        </div>
                      </div>
                    )
                  })}
                  {pairwiseResults.length === 0 && (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-gray-400">
                      Pairwise comparisons will appear here once judging is complete.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function LiveLog({ logs }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const { promptGroups, globalLogs } = useMemo(() => {
    const promptMap = new Map()
    const globals = []

    for (const log of logs) {
      if (!log?.prompt_hash) {
        if (log.type === 'postprocess_started' || log.type === 'postprocess_done') {
          globals.push(log)
        }
        continue
      }

      if (!promptMap.has(log.prompt_hash)) {
        promptMap.set(log.prompt_hash, {
          promptHash: log.prompt_hash,
          meta: {
            prompt_hash: log.prompt_hash,
            prompt_index: log.prompt_index,
            total: log.total,
            use_case: log.use_case,
            prompt_complexity: log.prompt_complexity,
            clarity: log.clarity,
          },
          steps: new Set(),
          selectedModels: [],
          progressLogs: [],
          failedLogs: [],
          pairwiseResults: [],
          sortIndex: typeof log.prompt_index === 'number' ? log.prompt_index : Number.MAX_SAFE_INTEGER,
        })
      }

      const group = promptMap.get(log.prompt_hash)
      group.meta = {
        ...group.meta,
        prompt_index: log.prompt_index ?? group.meta.prompt_index,
        total: log.total ?? group.meta.total,
        use_case: log.use_case ?? group.meta.use_case,
        prompt_complexity: log.prompt_complexity ?? group.meta.prompt_complexity,
        clarity: log.clarity ?? group.meta.clarity,
      }
      if (typeof log.prompt_index === 'number') {
        group.sortIndex = Math.min(group.sortIndex, log.prompt_index)
      }

      if (log.type === 'prompt_step' && log.step) {
        group.steps.add(log.step)
      }
      if (log.type === 'models_selected') {
        group.steps.add('models_selected')
        group.selectedModels = log.selected_models || []
      }
      if (log.type === 'progress') {
        group.progressLogs.push(log)
      }
      if (log.type === 'model_failed') {
        group.failedLogs.push(log)
      }
      if (log.type === 'pairwise_result') {
        group.pairwiseResults.push(log)
      }
    }

    const groups = Array.from(promptMap.values()).sort((a, b) => a.sortIndex - b.sortIndex)
    return { promptGroups: groups, globalLogs: globals }
  }, [logs])

  return (
    <div className="relative group">
      <div className="mb-4 flex items-center gap-2">
        <Terminal className="h-5 w-5 text-blue-500" />
        <label className="text-sm font-semibold text-gray-300">Live Pipeline View</label>
      </div>

      <div className="relative">
        <div className="absolute -inset-1 bg-gradient-to-r from-blue-600/10 via-indigo-600/10 to-blue-600/10 blur-3xl opacity-50 transition-opacity duration-1000" />
        <div className="relative h-[32rem] min-h-[420px] space-y-4 overflow-x-hidden overflow-y-auto rounded-2xl border border-gray-800/80 bg-[#030712]/80 p-6 font-mono text-[11px] shadow-2xl backdrop-blur-3xl custom-scrollbar">
          <AnimatePresence initial={false}>
            {globalLogs.map((log, index) => (
              <motion.div
                key={`${log.type}-${log.stage}-${index}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <GlobalStageCard log={log} />
              </motion.div>
            ))}
            {promptGroups.map((group, index) => (
              <motion.div
                key={group.promptHash}
                initial={{ opacity: 0, y: 10, scale: 0.99 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
              >
                <PromptCard group={group} defaultOpen={index === promptGroups.length - 1} />
              </motion.div>
            ))}
          </AnimatePresence>
          {promptGroups.length === 0 && globalLogs.length === 0 && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 text-sm text-gray-400">
              Telemetry for each prompt will appear here once the job starts.
            </div>
          )}
          <div ref={bottomRef} className="h-4" />
        </div>
      </div>
    </div>
  )
}
