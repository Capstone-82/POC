import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, MessageSquare, Briefcase, Cpu, ArrowRight, Loader2, Sparkles, AlertCircle, History, Info } from 'lucide-react'
import RecommendationOutput from '../components/RecommendationOutput'
import { getRecommendation } from '../api/inference'

const USE_CASES = [
  'General Chat', 'Technical Coding', 'Logical Reasoning', 'RAG (Pinecone/Weaviate)', 
  'Bulk Summarization', 'Fast Extraction', 'Complex Reasoning', 'Vision/OCR', 
  'Multi-modal Analysis', 'Agentic Workflows'
]

const MODELS = [
  'GPT-4o', 'GPT-4o Mini',
  'Gemini 2.5 Pro', 'Gemini 2.5 Flash', 'Gemini 2.5 Flash-Lite',
  'Claude 3.5 Sonnet', 'Claude 3 Haiku',
  'DeepSeek R1 (Bedrock)', 'DeepSeek V3',
  'Llama 3.3 70b', 'Llama 3.1 405b',
  'Mistral Large', 'Mistral Small', 'Pixtral Large',
  'Amazon Nova Pro', 'Amazon Nova Lite'
]

export default function Inference() {
  const [prompt, setPrompt] = useState('')
  const [useCase, setUseCase] = useState('')
  const [currentModel, setCurrentModel] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await getRecommendation({ 
        prompt, 
        use_case: useCase.toLowerCase(), 
        current_model: currentModel 
      })
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const canSubmit = prompt.trim() && useCase && currentModel

  return (
    <div className="max-w-5xl mx-auto px-8 py-12 space-y-12 pb-32">
      
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4 max-w-2xl"
      >
        <div className="flex items-center gap-2 px-3 py-1 bg-indigo-600/10 border border-indigo-500/20 rounded-full w-fit">
           <Zap className="w-3.5 h-3.5 text-indigo-500" />
           <span className="text-[10px] font-black uppercase text-indigo-500 tracking-widest">Inference Optimizer</span>
        </div>
        <h1 className="text-4xl font-black text-white tracking-tight">
          Intelligent <span className="gradient-text">Router</span> & Advise
        </h1>
        <p className="text-gray-400 font-medium leading-relaxed">
          Input your intent and current stack. Our recommendation engine parses benchmarks to find the 
          most cost-effective, high-accuracy path forward.
        </p>
      </motion.div>

      <div className="flex flex-col lg:flex-row gap-12">
        {/* Form Column */}
        <div className="flex-1 space-y-8">
           <div className="glass-card rounded-2xl p-8 border border-white/5 space-y-8 shadow-2xl">
             
             {/* Textarea */}
             <div className="space-y-3 group">
               <div className="flex items-center gap-2">
                 <MessageSquare className="w-4 h-4 text-indigo-500" />
                 <label className="text-xs font-black text-gray-400 uppercase tracking-widest">Target Prompt</label>
               </div>
               <textarea
                 value={prompt}
                 onChange={e => setPrompt(e.target.value)}
                 rows={6}
                 placeholder="Type or paste the prompt you want to optimize..."
                 className="w-full bg-gray-950/40 border border-gray-800 rounded-xl px-5 py-4
                            text-white text-sm placeholder-gray-700 resize-none
                            focus:outline-none focus:border-indigo-500/50 focus:ring-4 focus:ring-indigo-900/10
                            transition-all duration-300"
               />
             </div>

             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
               {/* Use Case */}
               <div className="space-y-3 group">
                 <div className="flex items-center gap-2">
                    <Briefcase className="w-4 h-4 text-indigo-500" />
                    <label className="text-xs font-black text-gray-400 uppercase tracking-widest">Operational Domain</label>
                 </div>
                 <div className="relative">
                   <select
                     value={useCase}
                     onChange={e => setUseCase(e.target.value)}
                     className="w-full bg-gray-950/40 border border-gray-800 rounded-xl px-5 py-3
                                text-white text-sm font-medium focus:outline-none focus:border-indigo-500/50
                                transition-all appearance-none cursor-pointer"
                   >
                     <option value="" className="bg-gray-900">Select Domain...</option>
                     {USE_CASES.map(u => (
                       <option key={u} value={u} className="bg-gray-900">{u}</option>
                     ))}
                   </select>
                   <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-600">
                      <ArrowRight className="w-4 h-4" />
                   </div>
                 </div>
               </div>

               {/* Current Model */}
               <div className="space-y-3 group">
                 <div className="flex items-center gap-2">
                    <History className="w-4 h-4 text-indigo-500" />
                    <label className="text-xs font-black text-gray-400 uppercase tracking-widest">Baseline Model</label>
                 </div>
                 <div className="relative">
                   <select
                     value={currentModel}
                     onChange={e => setCurrentModel(e.target.value)}
                     className="w-full bg-gray-950/40 border border-gray-800 rounded-xl px-5 py-3
                                text-white text-sm font-medium focus:outline-none focus:border-indigo-500/50
                                transition-all appearance-none cursor-pointer"
                   >
                     <option value="" className="bg-gray-900">Baseline Target...</option>
                     {MODELS.map(m => (
                       <option key={m} value={m} className="bg-gray-900">{m}</option>
                     ))}
                   </select>
                   <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-gray-600">
                      <ArrowRight className="w-4 h-4" />
                   </div>
                 </div>
               </div>
             </div>

             <button
               onClick={handleSubmit}
               disabled={!canSubmit || loading}
               className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800
                          disabled:text-gray-600 text-white text-xs font-black uppercase tracking-widest rounded-xl
                          transition-all duration-300 shadow-xl shadow-indigo-900/40 active:scale-[0.98]
                          flex items-center justify-center gap-3 group"
             >
               {loading ? (
                 <>
                   <Loader2 className="w-4 h-4 animate-spin" />
                   Running Classifier Engine...
                 </>
               ) : (
                 <>
                   Synthesize Recommendation
                   <Sparkles className="w-4 h-4 group-hover:scale-125 transition-transform" />
                 </>
               )}
             </button>
           </div>
        </div>

        {/* Output Column */}
        <div className="lg:w-1/3 xl:w-[400px] flex flex-col gap-6">
           <div className="flex items-center gap-3 px-4 py-3 bg-white/5 border border-white/5 rounded-xl">
              <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
                 <Info className="w-4 h-4 text-indigo-400" />
              </div>
              <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wider leading-relaxed">
                Analysis utilizes current snapshot from <strong>Production Benchmarks</strong> to calculate deltas.
              </p>
           </div>
           
           <AnimatePresence mode="wait">
             {loading && (
               <motion.div 
                 initial={{ opacity: 0, scale: 0.98 }}
                 animate={{ opacity: 1, scale: 1 }}
                 exit={{ opacity: 0, scale: 0.98 }}
                 className="flex-1 flex flex-col items-center justify-center gap-4 bg-indigo-600/5 border-2 border-dashed border-indigo-500/20 rounded-2xl p-12 text-center"
               >
                 <div className="relative">
                   <div className="w-16 h-16 border-4 border-indigo-900 rounded-full animate-spin-slow" />
                   <div className="absolute top-0 w-16 h-16 border-t-4 border-indigo-400 rounded-full animate-spin" />
                   <Cpu className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-indigo-400" />
                 </div>
                 <div className="space-y-1">
                   <h3 className="text-white font-black text-sm uppercase tracking-widest">Neural Routing...</h3>
                   <p className="text-gray-500 text-xs font-medium">Checking complexity against benchmark store</p>
                 </div>
               </motion.div>
             )}

             {result && <RecommendationOutput data={result} />}

             {error && (
               <motion.div 
                 initial={{ opacity: 0, x: 10 }}
                 animate={{ opacity: 1, x: 0 }}
                 className="bg-red-950/20 border border-red-500/20 rounded-2xl px-6 py-5 flex items-center gap-4"
               >
                 <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
                 <p className="text-red-400 text-xs font-bold font-mono tracking-tight">{error}</p>
               </motion.div>
             )}

             {!loading && !result && !error && (
               <div className="flex-1 flex flex-col items-center justify-center gap-4 bg-gray-950/40 border-2 border-dashed border-gray-900 rounded-2xl p-12 text-center grayscale opacity-40">
                 <Zap className="w-12 h-12 text-gray-700" />
                 <p className="text-gray-500 text-xs font-black uppercase tracking-widest">Awaiting Analysis</p>
               </div>
             )}
           </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
