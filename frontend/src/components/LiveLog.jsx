import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Terminal, Timer, Coins, Scale, Brain, ArrowUpRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export default function LiveLog({ logs }) {
  const bottomRef = useRef(null)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const filteredLogs = filter === 'all' 
    ? logs 
    : logs.filter(l => l.prompt_complexity === filter)

  return (
    <div className="relative group">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-blue-500" />
          <label className="text-sm font-semibold text-gray-300">
            Real-time Telemetry
          </label>
        </div>
        
        <div className="flex items-center gap-1.5 p-1 bg-white/5 rounded-lg border border-white/5">
          {['all', 'low', 'mid', 'high'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded-md text-[10px] uppercase font-bold tracking-widest transition-all duration-300 ${
                filter === f
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="relative">
        {/* Glow behind terminal */}
        <div className="absolute -inset-1 bg-gradient-to-r from-blue-600/10 via-indigo-600/10 to-blue-600/10 blur-3xl opacity-50 group-hover:opacity-100 transition-opacity duration-1000" />
        
        <div className="relative bg-[#030712]/80 border border-gray-800/80 rounded-2xl p-6
                        h-96 min-h-[400px] overflow-y-auto font-mono text-[11px] space-y-3
                        backdrop-blur-3xl shadow-2xl overflow-x-hidden custom-scrollbar">
          
          <AnimatePresence initial={false}>
            {filteredLogs.map((log, i) => (
              <motion.div
                key={`${log.model_id}-${log.prompt_index}-${i}`}
                initial={{ opacity: 0, x: -10, scale: 0.98 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                className="flex flex-col gap-2 p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/[0.08] hover:border-white/10 transition-colors group/item"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="px-2 py-0.5 bg-blue-600/20 text-blue-400 rounded-md font-bold tracking-tighter border border-blue-500/20">
                      {String(log.prompt_index).padStart(2, '0')}/{log.total}
                    </div>
                    <div className="flex items-center gap-1.5 text-white font-bold text-sm tracking-tight capitalize group-hover/item:text-blue-400 transition-colors">
                       {log.model_id}
                       <ArrowUpRight className="w-3 h-3 text-white/20 group-hover/item:text-blue-400" />
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded-full text-[9px] font-black uppercase tracking-widest ${
                      log.prompt_complexity === 'high' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                      log.prompt_complexity === 'mid' ? 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20' :
                      'bg-green-500/10 text-green-400 border border-green-500/20'
                    }`}>
                      {log.prompt_complexity}
                    </span>
                    <CheckCircle2 className="w-3.5 h-3.5 text-green-500/80 drop-shadow-lg" />
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4 pl-12">
                   <div className="flex items-center gap-2 group/metric">
                      <div className="p-1.5 rounded-lg bg-indigo-500/5 border border-indigo-500/10 group-hover/metric:bg-indigo-500/20 transition-colors">
                        <Scale className="w-3 h-3 text-indigo-400" />
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest">Quality</span>
                        <span className="text-white font-black">{log.accuracy_score}%</span>
                      </div>
                   </div>
                   
                   <div className="flex items-center gap-2 group/metric">
                      <div className="p-1.5 rounded-lg bg-orange-500/5 border border-orange-500/10 group-hover/metric:bg-orange-500/20 transition-colors">
                        <Timer className="w-3 h-3 text-orange-400" />
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest">Latency</span>
                        <span className="text-white font-black">{log.latency_ms}ms</span>
                      </div>
                   </div>

                   <div className="flex items-center gap-2 group/metric">
                      <div className="p-1.5 rounded-lg bg-green-500/5 border border-green-500/10 group-hover/metric:bg-green-500/20 transition-colors">
                        <Coins className="w-3 h-3 text-green-400" />
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest">Est. Cost</span>
                        <span className="text-white font-black">${log.cost?.toFixed(4)}</span>
                      </div>
                   </div>

                   <div className="flex items-center gap-2 group/metric">
                      <div className="p-1.5 rounded-lg bg-blue-500/5 border border-blue-500/10 group-hover/metric:bg-blue-500/20 transition-colors">
                        <Brain className="w-3 h-3 text-blue-400" />
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest">Provider</span>
                        <span className="text-white font-black">{log.provider || 'N/A'}</span>
                      </div>
                   </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={bottomRef} className="h-4" />
        </div>
        
        {/* Scanline Effect */}
        <div className="absolute inset-x-0 bottom-0 h-1/2 pointer-events-none bg-gradient-to-t from-[#030712]/40 to-transparent z-10 rounded-b-2xl" />
      </div>
    </div>
  )
}
