import { TrendingUp, Sparkles, AlertCircle, Quote, Cpu, Layers } from 'lucide-react'
import { motion } from 'framer-motion'

export default function RecommendationOutput({ data }) {
  const fmt = (val, unit = '', label = '') => {
    const isNumber = typeof val === 'number'
    const sign = isNumber && val > 0 ? '+' : ''
    // Specific logic for latency delta vs accuracy delta
    const isGood = label.toLowerCase().includes('accuracy') ? val >= 0 : val <= 0
    const colorClass = isGood ? 'text-green-400 bg-green-500/10 border-green-500/20' : 'text-red-400 bg-red-500/10 border-red-500/20'
    
    return (
      <div className={`px-3 py-1.5 rounded-lg border flex flex-col items-center justify-center gap-1 ${colorClass}`}>
         <span className="text-[10px] font-bold uppercase tracking-widest text-white/40">{label}</span>
         <span className="font-mono font-black text-xs">{sign}{val}{unit}</span>
      </div>
    )
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative group mt-8"
    >
      <div className="absolute -inset-1 bg-gradient-to-r from-blue-600/20 via-indigo-600/20 to-blue-600/20 blur-2xl group-hover:opacity-100 transition-opacity duration-1000 opacity-70" />
      
      <div className="relative bg-[#0f172a]/40 border border-white/10 rounded-2xl p-8 backdrop-blur-3xl shadow-2xl space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-yellow-400 animate-pulse" />
            <h3 className="text-lg font-bold text-white tracking-tight uppercase">Recommendation Analysis</h3>
          </div>
          <div className="flex items-center gap-3">
             <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-500/10 border border-yellow-500/20">
                <Layers className="w-3.5 h-3.5 text-yellow-500" />
                <span className="text-[10px] font-black uppercase text-yellow-500 tracking-widest">{data.complexity}Complexity</span>
             </div>
             <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20">
                <Cpu className="w-3.5 h-3.5 text-blue-500" />
                <span className="text-[10px] font-black uppercase text-blue-500 tracking-widest">Score: {data.quality_score}</span>
             </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Models */}
          <div className="space-y-4">
             <div className="flex flex-col gap-2">
                <span className="text-[10px] font-bold uppercase text-gray-500 tracking-widest">Comparison Point</span>
                <div className="px-5 py-4 rounded-xl bg-gray-950/40 border border-gray-800/80 flex items-center justify-between group/model transition-colors hover:border-white/10">
                   <span className="text-sm font-bold text-gray-400">{data.current_model}</span>
                   <span className="text-[10px] uppercase font-black px-2 py-0.5 rounded bg-gray-900 border border-gray-800 text-gray-600">Current</span>
                </div>
             </div>

             <div className="flex flex-col gap-2">
                <span className="text-[10px] font-bold uppercase text-blue-500 tracking-widest">Optimal Choice</span>
                <div className="px-5 py-4 rounded-xl bg-blue-600/5 border border-blue-500/20 flex items-center justify-between group/model shadow-lg shadow-blue-950/20 animate-in fade-in zoom-in duration-1000">
                   <span className="text-lg font-black text-white tracking-tight">{data.recommended_model}</span>
                   <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-blue-600 shadow-lg shadow-blue-900/40 border border-blue-400/50">
                      <TrendingUp className="w-3 h-3 text-white" />
                      <span className="text-[10px] uppercase font-black text-white">Recommended</span>
                   </div>
                </div>
             </div>
          </div>

          {/* Deltas */}
          <div className="flex flex-col justify-end gap-3">
             <div className="grid grid-cols-3 gap-3">
               {fmt(data.accuracy_delta, '%', 'Accuracy')}
               {fmt(data.cost_delta, '%', 'Cost')}
               {fmt(data.latency_delta, 'ms', 'Latency')}
             </div>
             <p className="text-[10px] text-gray-500 font-medium px-1 text-center italic">*Deltas relative to current baseline benchmark data</p>
          </div>
        </div>

        {/* Reason */}
        <div className="relative pt-6 border-t border-gray-800/80">
          <div className="absolute -top-3 left-4 px-2 bg-[#0f172a] flex items-center gap-1.5">
             <Quote className="w-3 h-3 text-blue-500" />
             <span className="text-[10px] font-bold uppercase text-gray-400 tracking-widest">Decision Rationale</span>
          </div>
          <p className="text-sm leading-relaxed text-gray-300 font-medium italic pl-4 border-l-2 border-blue-600/30">
            "{data.reason}"
          </p>
        </div>

        {/* Action Button */}
        <div className="flex justify-end pt-2">
           <button className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 transition-all rounded-lg text-xs font-bold uppercase tracking-widest text-white shadow-xl shadow-blue-900/40 group/btn">
              Acknowledge & Apply
              <TrendingUp className="w-4 h-4 transition-transform group-hover/btn:translate-x-1" />
           </button>
        </div>
      </div>
    </motion.div>
  )
}
