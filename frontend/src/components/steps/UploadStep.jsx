// frontend/src/components/steps/UploadStep.jsx
import { Upload, FolderOpen, ChevronDown, ChevronUp } from 'lucide-react'
import ImageUpload from '../ImageUpload'  // ← YOUR ACTUAL COMPONENT
import { useState } from 'react'

export default function UploadStep({ 
  imagePreview, 
  onImageUpload, 
  onNavigateToGallery,
  isProcessing 
}) {
  const [showDetectableObjects, setShowDetectableObjects] = useState(false);

  // Organized detection classes by category
  const detectionCategories = {
    animals: {
      label: 'Animals',
      items: [
        { name: 'Bear', accuracy: 98.9 },
        { name: 'Birds', accuracy: 96.4 },
        { name: 'Camel', accuracy: 95.5 },
        { name: 'Cat', accuracy: 95.4 },
        { name: 'Horse', accuracy: 92.8 },
        { name: 'Dog', accuracy: 87.8 },
        { name: 'Sheep', accuracy: 87.6 },
        { name: 'Cow', accuracy: 85.6 },
        { name: 'Elephant', accuracy: 81.1 },
      ]
    },
    vehicles: {
      label: 'Vehicles',
      items: [
        { name: 'Drones', accuracy: 94.3 },
        { name: 'Bus', accuracy: 88.0 },
        { name: 'Car', accuracy: 86.5 },
        { name: 'Airplane', accuracy: 84.8 },
        { name: 'Bicycle', accuracy: 73.1 },
      ]
    },
    human: {
      label: 'Human',
      items: [
        { name: 'Person', accuracy: 97.1 },
      ]
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 90) return 'text-emerald-600';
    if (accuracy >= 80) return 'text-blue-600';
    return 'text-amber-600';
  };

  const totalObjects = Object.values(detectionCategories).reduce((sum, cat) => sum + cat.items.length, 0);

  return (
    <div className="max-w-4xl mx-auto px-4">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Upload Your Query Image
        </h2>
        <p className="text-gray-600">
          AI-powered object detection • 15 categories • 89.7% accuracy
        </p>
      </div>

      {!imagePreview ? (
        <div className="space-y-6">
          {/* Upload Zone - YOUR ACTUAL COMPONENT */}
          <ImageUpload onImageUpload={onImageUpload} />

          {/* Divider */}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-3 bg-white text-gray-500">or</span>
            </div>
          </div>

          {/* Gallery Button */}
          <button
            onClick={onNavigateToGallery}
            className="w-full flex items-center justify-center gap-2 px-5 py-3.5 
                     border-2 border-dashed border-gray-300 rounded-lg
                     hover:border-blue-400 hover:bg-blue-50/50 transition-all group"
          >
            <FolderOpen className="w-5 h-5 text-gray-500 group-hover:text-blue-500" />
            <span className="text-gray-700 font-medium group-hover:text-blue-600">
              Choose from Gallery
            </span>
          </button>

          {/* Detectable Objects - Collapsible Section */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            {/* Clickable Header */}
            <button
              onClick={() => setShowDetectableObjects(!showDetectableObjects)}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold text-gray-900">
                  Detectable Objects ({totalObjects})
                </span>
                <span className="text-xs text-gray-500">
                  YOLOv8n • 89.7% avg
                </span>
              </div>
              {showDetectableObjects ? (
                <ChevronUp className="w-4 h-4 text-gray-500" />
              ) : (
                <ChevronDown className="w-4 h-4 text-gray-500" />
              )}
            </button>

            {/* Expandable Content */}
            {showDetectableObjects && (
              <div className="px-4 pb-4 bg-gray-50 border-t border-gray-200">
                <div className="space-y-4 pt-4">
                  {Object.entries(detectionCategories).map(([key, category]) => (
                    <div key={key}>
                      {/* Category Header */}
                      <div className="flex items-baseline gap-2 mb-2">
                        <h4 className="text-xs font-bold text-gray-700 uppercase tracking-wide">
                          {category.label}
                        </h4>
                        <span className="text-xs text-gray-500">
                          ({category.items.length})
                        </span>
                      </div>

                      {/* Category Items - Compact List */}
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {category.items.map((item) => (
                          <div
                            key={item.name}
                            className="flex items-center justify-between bg-white rounded px-3 py-2 border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all"
                          >
                            <span className="text-sm font-medium text-gray-800">
                              {item.name}
                            </span>
                            <span className={`text-xs font-bold ${getAccuracyColor(item.accuracy)}`}>
                              {Math.round(item.accuracy)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Legend */}
                <div className="flex items-center justify-center gap-4 mt-4 pt-3 border-t border-gray-200">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                    <span className="text-xs text-gray-600">≥90%</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-xs text-gray-600">80-89%</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                    <span className="text-xs text-gray-600">70-79%</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        // Image preview
        <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
          <div className="max-w-2xl mx-auto">
            <img 
              src={imagePreview} 
              alt="Uploaded" 
              className="w-full rounded-lg shadow-md"
            />
          </div>

          {isProcessing && (
            <div className="mt-6 flex items-center justify-center gap-3 bg-blue-50 rounded-lg p-3">
              <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
              <span className="text-blue-700 font-medium text-sm">Processing image...</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}