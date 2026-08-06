import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Training from './pages/Training'
import Inference from './pages/Inference'
import ClarityLabeling from './pages/ClarityLabeling'
import ProfilingRouting from './pages/ProfilingRouting'
import ModelRouting from './pages/ModelRouting'
import { motion } from 'framer-motion'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#030712] text-white selection:bg-blue-600/30 selection:text-white relative overflow-x-hidden">
        
        {/* Global Background Elements */}
        <div className="fixed inset-0 pointer-events-none z-0">
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full animate-pulse-slow" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/10 blur-[120px] rounded-full animate-pulse-slow" />
          <div className="absolute top-[30%] right-[10%] w-[20%] h-[20%] bg-purple-600/5 blur-[100px] rounded-full" />
          
          {/* Grain Effect Overlay */}
          <div className="absolute inset-0 opacity-[0.06] pointer-events-none bg-[radial-gradient(circle_at_1px_1px,rgba(148,163,184,0.22)_1px,transparent_0)] bg-[length:18px_18px]" />
        </div>

        <Navbar />
        
        <main className="relative z-10 px-4">
          <Routes>
            <Route path="/" element={<Navigate to="/training" replace />} />
            <Route 
              path="/training" 
              element={
                <motion.div 
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                  <Training />
                </motion.div>
              } 
            />
            <Route 
              path="/inference" 
              element={
                <motion.div 
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                  <Inference />
                </motion.div>
              } 
            />
            <Route
              path="/clarity"
              element={
                <motion.div
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                  <ClarityLabeling />
                </motion.div>
              }
            />
            <Route
              path="/profiling"
              element={
                <motion.div
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                  <ProfilingRouting />
                </motion.div>
              }
            />
            <Route
              path="/model-routing"
              element={
                <motion.div
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                  <ModelRouting />
                </motion.div>
              }
            />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

