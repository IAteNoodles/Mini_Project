import { useState } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { Bar } from 'react-chartjs-2'
import ReactMarkdown from 'react-markdown'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'
import clsx from 'clsx'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [explanationMode, setExplanationMode] = useState('SHAP')
  const [showExplanation, setShowExplanation] = useState(true)

  const [narrative, setNarrative] = useState(null)
  const [narrativeLoading, setNarrativeLoading] = useState(false)

  const handleAnalyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setNarrative(null)
    try {
      const response = await axios.post('/api/analyze', {
        text,
        explanation_mode: explanationMode
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during analysis.')
    } finally {
      setLoading(false)
    }
  }

  const handleExplain = async () => {
    if (!result) return
    setNarrativeLoading(true)
    try {
      const response = await axios.post('/api/explain', {
        text,
        label: result.label,
        explanation_mode: explanationMode,
        tokens: result.explanation.tokens,
        values: result.explanation.values
      })
      setNarrative(response.data.narrative)
    } catch (err) {
      setError('Failed to generate explanation.')
    } finally {
      setNarrativeLoading(false)
    }
  }

  return (
    <div className="w-full px-6 py-8">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-primary to-secondary text-white p-6 rounded-2xl shadow-lg mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">Bias Detector</h1>
        <p className="opacity-90">Analyze text for bias using AI-powered explanations.</p>
      </motion.div>

      <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/50 mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2 uppercase tracking-wider">Input Text</label>
        <textarea
          className="w-full p-4 rounded-lg border border-gray-200 focus:ring-2 focus:ring-primary/50 focus:border-primary outline-none transition-all min-h-[150px] resize-y bg-white/50 font-mono text-sm"
          placeholder="Paste article text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        
        <div className="flex flex-wrap items-center justify-between mt-4 gap-4">
          <div className="flex items-center gap-4">
            <div className="flex bg-gray-100 rounded-lg p-1">
              {['SHAP', 'LIME', 'Attention'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => setExplanationMode(mode)}
                  className={clsx(
                    "px-4 py-1.5 rounded-md text-sm font-medium transition-all",
                    explanationMode === mode 
                      ? "bg-white text-primary shadow-sm" 
                      : "text-gray-500 hover:text-gray-700"
                  )}
                >
                  {mode}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
              <input 
                type="checkbox" 
                checked={showExplanation} 
                onChange={(e) => setShowExplanation(e.target.checked)}
                className="rounded text-primary focus:ring-primary"
              />
              Show Visuals
            </label>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="bg-primary hover:bg-pink-600 text-white px-6 py-2.5 rounded-lg font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-primary/20"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
            {loading ? 'Analyzing...' : 'Detect Bias'}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-red-50 text-red-600 p-4 rounded-xl mb-6 flex items-center gap-3 border border-red-100"
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            {error}
          </motion.div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 lg:grid-cols-12 gap-6"
          >
            <div className="lg:col-span-7 space-y-6">
              <div className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-sm border border-white/50">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Verdict</h3>
                <div className="flex items-baseline gap-4">
                  <span className="text-4xl font-bold text-gray-900">{result.label}</span>
                  <span className="text-lg text-gray-500 font-medium">
                    {(result.score * 100).toFixed(2)}% confidence
                  </span>
                </div>
              </div>

              {showExplanation && result.heatmap_html && (
                 <div className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-sm border border-white/50">
                  <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">
                    Text Heatmap
                  </h3>
                  <div 
                    className="heatmap-container overflow-x-auto p-6 bg-white rounded-xl border border-gray-100 shadow-inner"
                    dangerouslySetInnerHTML={{ __html: result.heatmap_html }} 
                  />
                 </div>
              )}

              {showExplanation && result.explanation && result.explanation.heads && (
                <div className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-sm border border-white/50">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">
                        Attention Heads Analysis
                      </h3>
                      <p className="text-sm text-gray-500 mt-1">
                        Top 20 tokens. Darker color = Higher attention.
                      </p>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500 bg-gray-50 px-3 py-2 rounded-lg border border-gray-100">
                      <span>Low</span>
                      <div className="w-16 h-2 rounded-full bg-gradient-to-r from-blue-50 to-blue-600"></div>
                      <span>High</span>
                    </div>
                  </div>

                  <div className="overflow-x-auto pb-4">
                    <div className="min-w-[600px]">
                      <div className="grid gap-px bg-gray-200 border border-gray-200 rounded-lg overflow-hidden" style={{ gridTemplateColumns: `40px repeat(${result.explanation.tokens.length}, 1fr)` }}>
                        
                        {/* Header Row */}
                        <div className="bg-gray-50 p-2 flex items-end justify-center font-bold text-[10px] text-gray-400 uppercase">Head</div>
                        {result.explanation.tokens.map((token, idx) => (
                          <div key={idx} className="bg-gray-50 relative h-24 border-b border-gray-100">
                             <div className="absolute bottom-3 left-1/2 -translate-x-1/2 -rotate-45 origin-bottom text-[10px] text-gray-600 font-medium whitespace-nowrap">
                                {token}
                             </div>
                          </div>
                        ))}

                        {/* Data Rows */}
                        {result.explanation.heads.map((headData, headIdx) => (
                          <>
                            <div key={`h-${headIdx}`} className="bg-gray-50 text-[10px] font-bold text-gray-400 flex items-center justify-center">
                              {headIdx + 1}
                            </div>
                            {headData.map((val, colIdx) => {
                               const allValues = result.explanation.heads.flat();
                               const maxVal = Math.max(...allValues);
                               const intensity = maxVal > 0 ? (val / maxVal) : 0;
                               
                               return (
                                <div 
                                  key={`c-${headIdx}-${colIdx}`} 
                                  className="h-8 w-full relative group bg-white"
                                >
                                  <div 
                                    className="absolute inset-0 transition-all duration-300"
                                    style={{ backgroundColor: `rgba(37, 99, 235, ${intensity})` }}
                                  />
                                  
                                  {/* Tooltip */}
                                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-gray-900/95 backdrop-blur text-white text-xs py-1.5 px-3 rounded-lg shadow-xl whitespace-nowrap z-50 pointer-events-none border border-white/10">
                                    <div className="font-medium mb-0.5">Head {headIdx + 1} &rarr; <span className="text-blue-300">"{result.explanation.tokens[colIdx]}"</span></div>
                                    <div className="text-[10px] text-gray-400">Attention Score: {val.toFixed(4)}</div>
                                  </div>
                                </div>
                              )
                            })}
                          </>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {showExplanation && result.explanation && (
                <div className="bg-white/90 backdrop-blur rounded-xl p-6 shadow-sm border border-white/50">
                  <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">
                    {explanationMode} Highlights
                  </h3>
                  <div className="h-[300px]">
                    <Bar 
                      data={{
                        labels: result.explanation.tokens,
                        datasets: [{
                          label: 'Contribution',
                          data: result.explanation.values,
                          backgroundColor: result.explanation.values.map(v => v > 0 ? 'rgba(255, 0, 81, 0.7)' : 'rgba(0, 138, 255, 0.7)'),
                          borderColor: result.explanation.values.map(v => v > 0 ? 'rgba(255, 0, 81, 1)' : 'rgba(0, 138, 255, 1)'),
                          borderWidth: 1
                        }]
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        plugins: {
                          legend: { display: false }
                        },
                        scales: {
                          x: { 
                            grid: { display: false },
                            ticks: { font: { size: 10 } }
                          },
                          y: { 
                            grid: { display: false },
                            ticks: { font: { size: 11 } }
                          }
                        }
                      }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-5 space-y-6">
              <div className="bg-white/90 backdrop-blur rounded-xl shadow-sm border border-white/50 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Narrative Analysis</h3>
                  {!narrative && (
                    <button
                      onClick={handleExplain}
                      disabled={narrativeLoading}
                      className="text-xs bg-blue-50 text-blue-600 px-3 py-1.5 rounded-full font-medium hover:bg-blue-100 transition-colors disabled:opacity-50"
                    >
                      {narrativeLoading ? 'Generating...' : 'Explain with AI'}
                    </button>
                  )}
                </div>
                
                {narrative ? (
                  <div className="text-gray-700">
                    <ReactMarkdown
                      components={{
                        h3: ({node, ...props}) => (
                          <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mt-6 mb-3 flex items-center gap-2" {...props}>
                            <span className="w-1 h-3 bg-blue-500 rounded-full inline-block"></span>
                            {props.children}
                          </h3>
                        ),
                        p: ({node, ...props}) => <p className="text-sm text-gray-600 mb-4 leading-relaxed" {...props} />,
                        ul: ({node, ...props}) => <ul className="space-y-2 mb-6" {...props} />,
                        li: ({node, ...props}) => (
                          <li className="bg-gray-50/80 p-3 rounded-lg text-sm text-gray-700 border border-gray-100 shadow-sm flex gap-2">
                            <span className="text-blue-400 mt-0.5">•</span>
                            <span>{props.children}</span>
                          </li>
                        ),
                        strong: ({node, ...props}) => <span className="font-semibold text-gray-900 bg-blue-50/50 px-1 rounded" {...props} />,
                      }}
                    >
                      {narrative}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div className="text-sm text-gray-500 italic text-center py-8">
                    Click "Explain with AI" to generate a narrative analysis of the bias.
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
