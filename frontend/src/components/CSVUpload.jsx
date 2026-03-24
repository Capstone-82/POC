import { useDropzone } from 'react-dropzone'
import { Files, Upload, CheckCircle, FileX } from 'lucide-react'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

function cn(...inputs) {
  return twMerge(clsx(inputs))
}

export default function CSVUpload({ file, onFileChange }) {
  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    onDrop: (files) => onFileChange(files[0]),
  })

  return (
    <div className="group">
      <div className="flex items-center gap-2 mb-3">
        <Files className="w-5 h-5 text-blue-500" />
        <label className="text-sm font-semibold text-gray-300">
          Batch Dataset (CSV)
        </label>
      </div>

      <div
        {...getRootProps()}
        className={cn(
          "relative min-h-[160px] flex flex-col items-center justify-center p-8 text-center",
          "border-2 border-dashed rounded-2xl cursor-pointer overflow-hidden",
          "transition-all duration-500 ease-out",
          isDragActive 
            ? "border-blue-500 bg-blue-600/10 shadow-lg shadow-blue-500/20" 
            : "border-gray-800/80 bg-gray-950/40 hover:border-gray-600/80 hover:bg-gray-900/40 hover:shadow-2xl hover:shadow-blue-900/10"
        )}
      >
        <input {...getInputProps()} />

        {/* Dynamic Background */}
        <div 
          className={cn(
            "absolute inset-0 pointer-events-none opacity-0 transition-opacity duration-700",
            isDragActive && "opacity-100"
          )}
        >
          <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-blue-600/5 to-transparent shadow-2xl" />
        </div>

        {file ? (
          <div className="relative z-10 flex flex-col items-center gap-4 group">
            <div className="relative">
              <div className="w-16 h-16 rounded-2xl bg-green-500/10 border border-green-500/30 flex items-center justify-center shadow-lg shadow-green-950/40 group-hover:scale-110 transition-transform duration-500">
                 <CheckCircle className="w-8 h-8 text-green-400 drop-shadow-sm" />
              </div>
              <div className="absolute -top-1 -right-1 w-5 h-5 bg-green-500 rounded-full border-4 border-gray-950 animate-pulse" />
            </div>
            
            <div className="space-y-1">
              <p className="text-white text-base font-bold tracking-tight">
                {file.name}
              </p>
              <div className="flex items-center justify-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/5 mx-auto w-fit">
                <span className="text-gray-500 text-[10px] font-mono uppercase tracking-widest leading-none">
                  {(file.size / 1024).toFixed(1)} KB • CSV Ready
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="relative z-10 flex flex-col items-center gap-4">
             <div className={cn(
               "w-16 h-16 rounded-2xl flex items-center justify-center",
               "bg-blue-600/10 border border-blue-600/20 shadow-lg shadow-blue-950/40 group-hover:scale-110 transition-transform duration-500",
               isDragActive ? "bg-blue-500/20 scale-110 border-blue-500/40" : ""
             )}>
                <Upload className={cn(
                  "w-8 h-8 text-blue-500 transition-colors duration-500 group-hover:text-blue-400",
                  isDragActive && "animate-bounce text-blue-400"
                )} />
             </div>
             
             <div className="space-y-1.5 max-w-[280px]">
               <p className="text-gray-200 text-sm font-semibold tracking-tight">
                 {isDragActive ? "In-flight... Drop now" : "Drag & drop csv benchmark source"}
               </p>
               <p className="text-gray-500 text-xs leading-relaxed">
                 CSV must include <span className="font-mono text-blue-400 font-bold">"prompt"</span> column. 
                 Optional <span className="font-mono text-blue-400 font-bold">"accuracy"</span> column for user-provided scores (0-100).
               </p>
             </div>
          </div>
        )}
        
        {isDragReject && (
          <div className="absolute inset-0 bg-red-950/60 backdrop-blur-sm flex flex-col items-center justify-center gap-2 z-20">
            <FileX className="w-8 h-8 text-red-500" />
            <p className="text-red-200 text-sm font-bold">Unsupported file type</p>
          </div>
        )}
      </div>
    </div>
  )
}
