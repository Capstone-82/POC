import { NavLink } from 'react-router-dom'
import { Activity, Zap, BrainCircuit } from 'lucide-react'

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 glass-card mx-4 mt-4 rounded-xl px-8 py-4 flex items-center justify-between border border-white/10">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-900/40">
          <Zap className="w-5 h-5 text-white" />
        </div>
        <span className="text-white font-bold text-xl tracking-tight gradient-text">
          ModelMatrix
        </span>
      </div>
      
      <div className="flex items-center gap-2 p-1 bg-white/5 rounded-xl border border-white/10">
        <NavLink
          to="/training"
          className={({ isActive }) =>
            `px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-300 flex items-center gap-2 ${
              isActive
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`
          }
        >
          <Activity className="w-4 h-4" />
          Benchmark
        </NavLink>
        <NavLink
          to="/inference"
          className={({ isActive }) =>
            `px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-300 flex items-center gap-2 ${
              isActive
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`
          }
        >
          <Zap className="w-4 h-4" />
          Recommend
        </NavLink>
        <NavLink
          to="/clarity"
          className={({ isActive }) =>
            `px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-300 flex items-center gap-2 ${
              isActive
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/40'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`
          }
        >
          <BrainCircuit className="w-4 h-4" />
          Clarity
        </NavLink>
      </div>
      
      <div className="hidden md:flex items-center gap-4 text-xs font-medium text-gray-500 uppercase tracking-widest">
        v1.0.0
      </div>
    </nav>
  )
}
