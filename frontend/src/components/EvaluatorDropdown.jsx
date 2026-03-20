import { ChevronDown, BrainCircuit } from 'lucide-react'

const EVALUATORS = [
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash (Default)' },
  { value: 'gpt-4o', label: 'GPT-4o (GCP)' },
  { value: 'claude-3-5-sonnet', label: 'Claude 3.5 Sonnet' },
]

export default function EvaluatorDropdown({ value, onChange }) {
  return (
    <div className="relative">
      <div className="flex items-center gap-2 mb-3">
        <BrainCircuit className="w-5 h-5 text-blue-500" />
        <label className="text-sm font-semibold text-gray-300">
          AI Judge Evaluator
        </label>
      </div>
      
      <div className="relative group">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full max-w-sm bg-gray-900/60 border border-gray-700/50 rounded-xl
                     px-5 py-3 text-white text-sm font-medium focus:outline-none
                     focus:border-blue-500/50 focus:ring-4 focus:ring-blue-900/20
                     transition-all duration-300 appearance-none cursor-pointer
                     backdrop-blur-xl group-hover:border-gray-500/50"
        >
          {EVALUATORS.map(e => (
            <option key={e.value} value={e.value}>{e.label}</option>
          ))}
        </select>
        <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400 group-hover:text-white transition-colors">
          <ChevronDown className="w-4 h-4" />
        </div>
      </div>
      
      <p className="mt-2 text-xs text-gray-500 flex items-center gap-1.5 px-1">
        <span className="w-1 h-1 rounded-full bg-blue-500 animate-pulse" />
        Used to score benchmark responses & classify prompts
      </p>
    </div>
  )
}
