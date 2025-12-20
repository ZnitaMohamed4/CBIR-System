// frontend/src/components/FeatureViewer.jsx

import { X, Palette, Grid3x3, Shapes, Info } from 'lucide-react'
import { useState } from 'react'

function FeatureViewer({ features, onClose }) {
  const [activeTab, setActiveTab] = useState('color')

  if (!features) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl p-6 max-w-2xl w-full">
          <p className="text-center text-slate-600">No features available</p>
        </div>
      </div>
    )
  }

  const tabs = [
    { id: 'color', label: 'Color', icon: Palette },
    { id: 'texture', label: 'Texture', icon: Grid3x3 },
    { id: 'shape', label: 'Shape', icon: Shapes },
  ] 

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-slate-200 flex items-center justify-between bg-gradient-to-r from-blue-50 to-indigo-50">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Info className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-800">Extracted Features</h2>
              <p className="text-sm text-slate-600">Visual characteristics of detected object</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 transition-colors p-2 hover:bg-white rounded-lg"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-200 bg-slate-50">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center space-x-2 py-4 font-semibold transition-all ${
                  activeTab === tab.id
                    ? 'text-blue-600 bg-white border-b-2 border-blue-600'
                    : 'text-slate-600 hover:text-slate-800 hover:bg-slate-100'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'color' && <ColorFeatures features={features.color} />}
          {activeTab === 'texture' && <TextureFeatures features={features.texture} />}
          {activeTab === 'shape' && <ShapeFeatures features={features.shape} />}
        </div>
      </div>
    </div>
  )
}

function ColorFeatures({ features }) {
  return (
    <div className="space-y-6">
      {/* Dominant Colors */}
      {features.dominant_colors && features.dominant_colors.length > 0 && (
        <div>
          <h3 className="text-lg font-bold text-slate-800 mb-3">🎨 Dominant Colors</h3>
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <div className="flex space-x-3">
              {features.dominant_colors.map((color, idx) => (
                <div key={idx} className="flex-1 text-center">
                  <div 
                    className="w-full h-16 rounded-lg shadow-inner border border-slate-300"
                    style={{ backgroundColor: color.hex }}
                  />
                  <p className="text-xs text-slate-600 mt-2 font-mono">{color.hex}</p>
                  <p className="text-xs text-slate-500">{color.percentage}%</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Mean RGB Values</h3>
        <div className="grid grid-cols-3 gap-4">
          {features.mean_rgb && features.mean_rgb.map((value, idx) => (
            <div key={idx} className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <p className="text-sm text-slate-600 mb-1">{['Red', 'Green', 'Blue'][idx]}</p>
              <p className="text-2xl font-bold text-slate-800">{Math.round(value)}</p>
              <div className="mt-2 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full"
                  style={{
                    width: `${(value / 255) * 100}%`,
                    backgroundColor: ['#EF4444', '#10B981', '#3B82F6'][idx]
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">RGB Histogram</h3>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
          {features.histogram_rgb && features.histogram_rgb.length > 0 ? (
            <div className="flex items-end space-x-1 h-32">
              {(() => {
                // Find max value for scaling
                const maxValue = Math.max(...features.histogram_rgb)
                
                return features.histogram_rgb.slice(0, 48).map((value, idx) => {
                  // Scale to max value for better visualization
                  const heightPercent = maxValue > 0 ? (value / maxValue) * 100 : 0
                  
                  return (
                    <div
                      key={idx}
                      className="flex-1 rounded-t transition-all hover:opacity-100"
                      style={{
                        height: `${heightPercent}%`,
                        minHeight: value > 0 ? '2px' : '0px', // Ensure visible bars
                        backgroundColor: idx < 16 ? '#EF4444' : idx < 32 ? '#10B981' : '#3B82F6',
                        opacity: 0.8
                      }}
                      title={`Bin ${idx}: ${(value * 100).toFixed(2)}%`}
                    />
                  )
                })
              })()}
            </div>
          ) : (
            <p className="text-sm text-slate-500 italic">No histogram data available</p>
          )}
        </div>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Standard Deviation</h3>
        <div className="grid grid-cols-3 gap-4">
          {features.std_rgb && features.std_rgb.map((value, idx) => (
            <div key={idx} className="bg-slate-50 p-4 rounded-lg border border-slate-200">
              <p className="text-sm text-slate-600 mb-1">{['Red', 'Green', 'Blue'][idx]}</p>
              <p className="text-xl font-bold text-slate-800">{value.toFixed(2)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function TextureFeatures({ features }) {
  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-800">
          <strong>Texture features</strong> capture the surface characteristics and patterns in the image.
        </p>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Tamura Features</h3>
        <div className="space-y-4">
          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-slate-700">Coarseness</span>
              <span className="text-sm text-slate-500">{features.tamura_coarseness?.toFixed(4)}</span>
            </div>
            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500"
                style={{ width: `${Math.min(features.tamura_coarseness / 10, 1) * 100}%` }}
              />
            </div>
            <p className="text-xs text-slate-600 mt-1">Measures texture granularity (higher = coarser)</p>
          </div>

          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-slate-700">Contrast</span>
              <span className="text-sm text-slate-500">{features.tamura_contrast?.toFixed(4)}</span>
            </div>
            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500"
                style={{ width: `${Math.min(features.tamura_contrast / 50, 1) * 100}%` }}
              />
            </div>
            <p className="text-xs text-slate-600 mt-1">Measures intensity variation (higher = more contrast)</p>
          </div>

          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold text-slate-700">Directionality</span>
              <span className="text-sm text-slate-500">{features.tamura_directionality?.toFixed(4)}</span>
            </div>
            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500"
                style={{ width: `${Math.min(features.tamura_directionality / 5, 1) * 100}%` }}
              />
            </div>
            <p className="text-xs text-slate-600 mt-1">Measures edge directionality (higher = more directional)</p>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Gabor Features</h3>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
          <p className="text-sm text-slate-600">
            <strong>{features.gabor_features_count}</strong> Gabor filter responses extracted
          </p>
          <p className="text-xs text-slate-500 mt-2">
            Gabor filters detect edges and textures at different orientations and frequencies
          </p>
        </div>
      </div>

      {/* LBP Features */}
      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Local Binary Pattern (LBP)</h3>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-xs text-slate-500">LBP Mean</span>
              <p className="text-lg font-bold text-slate-800">{features.lbp_mean?.toFixed(4) || 'N/A'}</p>
            </div>
            <div>
              <span className="text-xs text-slate-500">LBP Std Dev</span>
              <p className="text-lg font-bold text-slate-800">{features.lbp_std?.toFixed(4) || 'N/A'}</p>
            </div>
          </div>
          {features.lbp_histogram && features.lbp_histogram.length > 0 && (
            <div>
              <span className="text-xs text-slate-500">LBP Histogram ({features.lbp_histogram.length} bins)</span>
              <div className="flex items-end space-x-1 h-16 mt-2">
                {features.lbp_histogram.map((value, idx) => (
                  <div
                    key={idx}
                    className="flex-1 bg-gradient-to-t from-teal-500 to-teal-400 rounded-t"
                    style={{
                      height: `${Math.min(value, 1) * 100}%`,
                      opacity: 0.8
                    }}
                    title={`Bin ${idx}: ${(value * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
            </div>
          )}
          <p className="text-xs text-slate-500">
            LBP captures micro-patterns in texture (rotation invariant)
          </p>
        </div>
      </div>
    </div>
  )
}

function ShapeFeatures({ features }) {
  return (
    <div className="space-y-6">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <p className="text-sm text-purple-800">
          <strong>Shape features</strong> capture the geometric structure and form of the object.
        </p>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Hu Moments</h3>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
          <p className="text-sm text-slate-600 mb-3">
            7 rotation, scale, and translation invariant shape descriptors
          </p>
          <div className="grid grid-cols-2 gap-3">
            {features.hu_moments && features.hu_moments.map((value, idx) => (
              <div key={idx} className="bg-white p-3 rounded border border-slate-200">
                <span className="text-xs text-slate-500">Moment {idx + 1}</span>
                <p className="text-sm font-mono font-semibold text-slate-800 mt-1">
                  {value.toFixed(6)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">HOG (Histogram of Oriented Gradients)</h3>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
          <p className="text-sm text-slate-600">
            <strong>{features.hog_features_count}</strong> HOG features extracted
          </p>
          <p className="text-xs text-slate-500 mt-2">
            HOG features capture edge directions and shapes, commonly used for object detection
          </p>
        </div>
      </div>

      {/* Contour Orientation Features */}
      <div>
        <h3 className="text-lg font-bold text-slate-800 mb-3">Contour Orientation</h3>
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-xs text-slate-500">Main Orientation</span>
              <p className="text-lg font-bold text-slate-800">{features.main_orientation?.toFixed(1) || 0}°</p>
            </div>
            <div>
              <span className="text-xs text-slate-500">Orientation Variance</span>
              <p className="text-lg font-bold text-slate-800">{features.orientation_variance?.toFixed(4) || 'N/A'}</p>
            </div>
          </div>
          {features.contour_orientation_hist && features.contour_orientation_hist.length > 0 && (
            <div>
              <span className="text-xs text-slate-500">Edge Orientation Histogram (0° - 180°)</span>
              <div className="flex items-end space-x-1 h-16 mt-2">
                {features.contour_orientation_hist.map((value, idx) => (
                  <div
                    key={idx}
                    className="flex-1 bg-gradient-to-t from-orange-500 to-amber-400 rounded-t"
                    style={{
                      height: `${Math.min(value, 1) * 100}%`,
                      opacity: 0.8
                    }}
                    title={`${idx * 10}° - ${(idx + 1) * 10}°: ${(value * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>0°</span>
                <span>90°</span>
                <span>180°</span>
              </div>
            </div>
          )}
          <p className="text-xs text-slate-500">
            Contour orientation captures the edge direction distribution from the most significant contour
          </p>
        </div>
      </div>
    </div>
  )
}

export default FeatureViewer