// frontend/src/components/steps/SearchConfigStep.jsx
import { useState } from 'react'
import { Info, ChevronDown, ChevronUp } from 'lucide-react'
import FeatureWeights from '../FeatureWeights'

export default function SearchConfigStep({ 
  onWeightsChange, 
  onSearch,
  selectedObject 
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const presets = [
    { name: 'Balanced', color: 33, texture: 33, shape: 34, description: 'Equal importance to all features' },
    { name: 'Color-Focused', color: 70, texture: 15, shape: 15, description: 'Prioritize color similarity' },
    { name: 'Texture-Focused', color: 15, texture: 70, shape: 15, description: 'Prioritize texture patterns' },
    { name: 'Shape-Focused', color: 15, texture: 15, shape: 70, description: 'Prioritize object shape' },
  ]

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Configure Search Parameters
        </h2>
        <p className="text-gray-600">
          Smart AI-powered similarity search with automatic feature optimization
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-8 space-y-6">
        
        {/* ✅ INFO BANNER */}
        <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg p-4 flex items-start space-x-3">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-blue-900 mb-1">
              Automatic Feature Optimization
            </p>
            <p className="text-sm text-blue-700">
              Feature weights are automatically optimized for each object class. 
              Cars prioritize shape, food prioritizes color, animals balance texture + color.
            </p>
          </div>
        </div>

        {/* ✅ QUICK PRESETS - READ ONLY */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center space-x-2">
            <span>Reference: Common Weight Distributions</span>
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {presets.map((preset) => (
              <div
                key={preset.name}
                className="p-4 border-2 border-gray-200 rounded-lg bg-gray-50 text-center"
              >
                <div className="font-semibold text-gray-900 mb-1">{preset.name}</div>
                <div className="text-xs font-mono text-blue-600 mb-2">
                  {preset.color}/{preset.texture}/{preset.shape}
                </div>
                <div className="text-xs text-gray-500">
                  {preset.description}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ✅ ADVANCED MODE TOGGLE */}
        <div className="border-t border-gray-200 pt-4">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <div className="flex items-center space-x-2">
              <span className="text-sm font-semibold text-gray-700">
                Advanced Mode: Custom Weight Override
              </span>
              <span className="px-2 py-0.5 bg-amber-100 text-amber-700 text-xs rounded-full font-semibold">
                Optional
              </span>
            </div>
            {showAdvanced ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </button>

          {/* ✅ COLLAPSIBLE CUSTOM WEIGHTS */}
          {showAdvanced && (
            <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
              <div className="flex items-start space-x-2 mb-4">
                <Info className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-700">
                  <strong>Warning:</strong> Manual weights override the automatic class-specific optimization. 
                  Only adjust if you have specific requirements.
                </p>
              </div>
              <FeatureWeights 
                onWeightsChange={onWeightsChange}
                onSearch={onSearch}
              />
            </div>
          )}
        </div>

        {/* ✅ SEARCH BUTTON (when Advanced Mode is hidden) */}
        {!showAdvanced && (
          <button 
            onClick={() => onSearch(null)}  // null = use automatic weights
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center justify-center space-x-3"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span className="text-lg">Search Similar Images</span>
          </button>
        )}
      </div>
    </div>
  )
}