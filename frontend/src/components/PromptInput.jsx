import { MessageSquareText } from 'lucide-react'

export default function PromptInput({ value, onChange }) {
  return (
    <div className="group">
      <div className="flex items-center gap-2 mb-3">
        <MessageSquareText className="w-5 h-5 text-blue-500" />
        <label className="text-sm font-semibold text-gray-300">
          Raw Prompt Intent
        </label>
      </div>
      
      <div className="relative">
        <div className="absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-blue-600/0 via-blue-600/50 to-blue-600/0 opacity-0 group-focus-within:opacity-100 transition-opacity duration-700 blur-sm -z-1" />
        <textarea
          value={value}
          onChange={e => onChange(e.target.value)}
          rows={6}
          placeholder="System prompt and user instructions here..."
          className="w-full bg-gray-950/60 border border-gray-800/80 rounded-2xl px-6 py-5
                     text-white text-sm placeholder-gray-600 resize-none
                     focus:outline-none focus:border-blue-500/50 focus:ring-4 focus:ring-blue-900/10
                     transition-all duration-300 backdrop-blur-3xl shadow-2xl relative z-10"
        />
        
        <div className="absolute right-4 bottom-4 flex items-center gap-2 z-20">
          <span className="text-[10px] font-mono font-medium text-gray-600 uppercase tracking-tighter">
            {value.length} chars
          </span>
        </div>
      </div>
    </div>
  )
}
