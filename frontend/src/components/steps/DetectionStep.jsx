// frontend/src/components/steps/DetectionStep.jsx
import { useState, useEffect } from 'react'
import { Target, CheckCircle2, Eye, Loader2, Palette, Waves, Shapes, AlertCircle, Sliders } from 'lucide-react'
import ImageWithBoundingBoxes from '../ImageWithBoundingBoxes'
import api from '../../services/api'

export default function DetectionStep({ 
  imagePreview, 
  detectedObjects, 
  selectedObjects,
  onToggleSelection,
  onClear,
  onNext,
  onViewFeatures,
  imageId
}) {
  const [extractedFeatures, setExtractedFeatures] = useState({})
  const [loadingFeatures, setLoadingFeatures] = useState({})
  const [featureErrors, setFeatureErrors] = useState({})
  const [confidenceThreshold, setConfidenceThreshold] = useState(50) // 50% default

  // Filter objects based on confidence threshold
  const filteredObjects = detectedObjects.filter(
    obj => (obj.confidence * 100) >= confidenceThreshold
  )

  // Auto-extract features for selected object
  useEffect(() => {
    if (selectedObjects.length > 0 && imageId) {
      const objectId = selectedObjects[0]
      console.log('🔍 Auto-extracting features for:', { imageId, objectId })
      
      if (!extractedFeatures[objectId] && !loadingFeatures[objectId]) {
        extractFeatures(objectId)
      }
    }
  }, [selectedObjects, imageId])

  const extractFeatures = async (objectId) => {
    console.log('📤 Extracting features...', { imageId, objectId })
    setLoadingFeatures(prev => ({ ...prev, [objectId]: true }))
    setFeatureErrors(prev => ({ ...prev, [objectId]: null }))
    
    try {
      // Validate inputs
      if (!imageId) {
        throw new Error('No image ID available')
      }
      
      if (objectId === undefined || objectId === null) {
        throw new Error('Invalid object ID')
      }

      // Extract features
      console.log('🔧 Calling extractFeatures API...')
      const extractResult = await api.extractFeatures(imageId, objectId)
      console.log('✅ Extract result:', extractResult)
      
      // Get features
      console.log('📥 Fetching features...')
      const result = await api.getFeatures(imageId, objectId)
      console.log('✅ Get features result:', result)
      
      setExtractedFeatures(prev => ({ ...prev, [objectId]: result.features }))
      console.log('✨ Features saved to state')
      
    } catch (error) {
      console.error('❌ Feature extraction failed:', error)
      setFeatureErrors(prev => ({ ...prev, [objectId]: error.message }))
    } finally {
      setLoadingFeatures(prev => ({ ...prev, [objectId]: false }))
    }
  }

  const renderFeatureSummary = (objectId) => {
    const features = extractedFeatures[objectId]
    const isLoading = loadingFeatures[objectId]
    const error = featureErrors[objectId]

    // Loading state
    if (isLoading) {
      return (
        <div className="mt-3 p-3 bg-blue-50 rounded-lg">
          <div className="flex items-center space-x-2 text-blue-600">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Extracting features...</span>
          </div>
        </div>
      )
    }

    // Error state
    if (error) {
      return (
        <div className="mt-3 p-3 bg-red-50 rounded-lg">
          <div className="flex items-start space-x-2 text-red-600">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <div className="text-xs">
              <div className="font-semibold">Failed to extract features</div>
              <div className="text-red-500 mt-1">{error}</div>
              <button
                onClick={() => extractFeatures(objectId)}
                className="mt-2 text-red-700 underline hover:text-red-800"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      )
    }

    // No features yet
    if (!features) {
      return (
        <button
          onClick={() => extractFeatures(objectId)}
          className="mt-3 w-full px-3 py-2 text-sm bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-lg transition-colors"
        >
          Extract Features
        </button>
      )
    }

    // Display feature summary
    const dominantColors = features.color?.dominant_colors?.slice(0, 3) || []

    return (
      <div className="mt-3 space-y-2">
        {/* Dominant Colors */}
        {dominantColors.length > 0 && (
          <div className="flex items-center space-x-2">
            <Palette className="w-4 h-4 text-gray-500" />
            <div className="flex space-x-1">
              {dominantColors.map((color, idx) => (
                <div
                  key={idx}
                  className="w-6 h-6 rounded-full border-2 border-white shadow"
                  style={{ backgroundColor: color.hex }}
                  title={`${color.hex} (${color.percentage}%)`}
                />
              ))}
            </div>
          </div>
        )}

        {/* Feature Stats */}
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-purple-50 p-2 rounded text-center">
            <Palette className="w-3 h-3 mx-auto mb-1 text-purple-600" />
            <div className="font-semibold text-purple-900">Color</div>
          </div>
          <div className="bg-orange-50 p-2 rounded text-center">
            <Waves className="w-3 h-3 mx-auto mb-1 text-orange-600" />
            <div className="font-semibold text-orange-900">Texture</div>
          </div>
          <div className="bg-green-50 p-2 rounded text-center">
            <Shapes className="w-3 h-3 mx-auto mb-1 text-green-600" />
            <div className="font-semibold text-green-900">Shape</div>
          </div>
        </div>

        {/* View Details Button */}
        <button
          onClick={onViewFeatures}
          className="w-full flex items-center justify-center space-x-1 text-sm text-blue-600 hover:text-blue-700 font-medium py-2 hover:bg-blue-50 rounded transition-colors"
        >
          <Eye className="w-4 h-4" />
          <span>View Full Features</span>
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Select Objects & View Features
        </h2>
        <p className="text-gray-600">
          {filteredObjects.length} object{filteredObjects.length !== 1 ? 's' : ''} detected
          {filteredObjects.length !== detectedObjects.length && (
            <span className="text-gray-400 ml-1">
              ({detectedObjects.length - filteredObjects.length} filtered by confidence)
            </span>
          )}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Image Preview - Larger */}
        <div className="col-span-2">
          <div className="bg-white rounded-xl shadow-lg p-4">
            <ImageWithBoundingBoxes 
              imageUrl={imagePreview} 
              detections={filteredObjects}
              onClear={onClear}
            />
          </div>
        </div>

        {/* Object List with Features */}
        <div className="col-span-1 space-y-4">
          {/* ✅ CONFIDENCE THRESHOLD FILTER */}
          <div className="bg-white rounded-xl shadow-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-gray-600" />
                <span className="text-sm font-semibold text-gray-700">
                  Confidence Filter
                </span>
              </div>
              <span className="text-sm font-bold text-blue-600">
                ≥ {confidenceThreshold}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
              className="w-full cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>

          {/* Object List */}
          <div className="bg-white rounded-xl shadow-lg p-6 sticky top-32">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-blue-600" />
              Detected Objects
            </h3>

            {filteredObjects.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <AlertCircle className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                <p className="font-semibold">No objects above threshold</p>
                <p className="text-sm mt-1">Lower the confidence filter to see more detections</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[500px] overflow-y-auto">
                {filteredObjects.map((obj) => (
                  <div
                    key={obj.id}
                    className={`
                      p-4 rounded-lg border-2 transition-all
                      ${selectedObjects.includes(obj.id)
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200'
                      }
                    `}
                  >
                    {/* Object Header - Clickable */}
                    <button
                      onClick={() => onToggleSelection(obj.id)}
                      className="w-full text-left"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-gray-900">
                            {obj.label}
                          </div>
                          <div className={`text-sm font-semibold ${
                            obj.confidence >= 0.8 ? 'text-green-600' :
                            obj.confidence >= 0.6 ? 'text-amber-600' :
                            'text-orange-600'
                          }`}>
                            {Math.round(obj.confidence * 100)}% confident
                          </div>
                        </div>
                        {selectedObjects.includes(obj.id) && (
                          <CheckCircle2 className="w-5 h-5 text-blue-600" />
                        )}
                      </div>
                    </button>

                    {/* Feature Summary - Only for selected */}
                    {selectedObjects.includes(obj.id) && renderFeatureSummary(obj.id)}
                  </div>
                ))}
              </div>
            )}

            <button
              onClick={onNext}
              disabled={selectedObjects.length === 0 || filteredObjects.length === 0}
              className="mt-6 w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Continue to Search →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}