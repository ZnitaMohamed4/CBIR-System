// ============================================
// src/components/SearchResults.jsx
// ============================================

import { Loader2 } from 'lucide-react'
import api from '../services/api'  // Add this import at top


function SearchResults({ results, isLoading }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="relative mx-auto mb-6 w-20 h-20">
            <Loader2 className="w-20 h-20 text-blue-600 animate-spin" />
            <div className="absolute inset-0 blur-xl bg-blue-400 opacity-30 animate-pulse"></div>
          </div>
          <p className="text-slate-700 font-bold text-xl mb-2">Searching Database...</p>
          <p className="text-slate-500 text-sm">Analyzing visual features and computing similarity</p>
        </div>
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="bg-gradient-to-br from-blue-100 to-indigo-100 w-32 h-32 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
            <span className="text-6xl">🔍</span>
          </div>
          <p className="text-2xl font-bold text-slate-800 mb-3">No Results Yet</p>
          <p className="text-slate-600">
            Upload a query image, select detected objects, and click <span className="font-semibold text-blue-600">"Search"</span> to find similar images
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-slate-600">
            Found <span className="font-bold text-blue-600">{results.length}</span> similar images
          </p>
        </div>
      </div>
      
      <div className="image-grid">
        {results.map((result) => (
          <div key={result.id} className="image-card bg-white shadow-md">

            <img 
              src={api.getImageUrl(result.filename)}
              alt={`${result.className} - ${result.similarity}% match`}
              className="w-full object-cover"  // Removed h-56, now controlled by CSS
              onError={(e) => {
                e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23ddd" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999"%3EImage%3C/text%3E%3C/svg%3E'
              }}
            />
            
            <div className="overlay">
              <div className="w-full space-y-2">
                <div className={`badge px-3 py-2 text-sm font-bold shadow-lg ${
                  result.similarity >= 90 ? 'similarity-high' : 
                  result.similarity >= 70 ? 'similarity-medium' : 'similarity-low'
                }`}>
                  {result.similarity}% Match
                </div>
                <p className="text-white font-semibold text-sm">
                  {result.className} ({result.confidence}%)
                </p>
                <p className="text-white text-xs opacity-80">
                  Object #{result.objectId + 1}
                </p>
              </div>
            </div>
            
            {/* Corner badge */}
            <div className={`absolute top-3 right-3 px-2.5 py-1 rounded-lg text-xs font-bold text-white shadow-lg ${
              result.similarity >= 90 ? 'bg-emerald-500' : 
              result.similarity >= 70 ? 'bg-amber-500' : 'bg-rose-500'
            }`}>
              {result.similarity}%
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SearchResults