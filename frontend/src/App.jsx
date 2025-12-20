// frontend/src/App.jsx

import { useState } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import DetectedObjects from './components/DetectedObjects'
import FeatureWeights from './components/FeatureWeights'
import SearchResults from './components/SearchResults'
import ImageWithBoundingBoxes from './components/ImageWithBoundingBoxes'
import api from './services/api'
import Gallery from './pages/Gallery'
import FeatureViewer from './components/FeatureViewer'
import Toast from './components/Toast'
import { Info } from 'lucide-react'

// Import new step components
import WorkflowTabs from './components/steps/WorkflowTabs'
import UploadStep from './components/steps/UploadStep'
import DetectionStep from './components/steps/DetectionStep'
import SearchConfigStep from './components/steps/SearchConfigStep'
import ResultsStep from './components/steps/ResultsStep'

function App() {
  // Existing state
  const [uploadedImage, setUploadedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [detectedObjects, setDetectedObjects] = useState([])
  const [selectedObjects, setSelectedObjects] = useState([])
  const [searchResults, setSearchResults] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const [featureWeights, setFeatureWeights] = useState({ color: 40, texture: 30, shape: 30 })
  const [imageId, setImageId] = useState(null)
  const [currentPage, setCurrentPage] = useState('upload')
  const [showFeatures, setShowFeatures] = useState(false)
  const [currentFeatures, setCurrentFeatures] = useState(null)
  const [toast, setToast] = useState(null)

  // NEW: Step workflow state
  const [currentStep, setCurrentStep] = useState(1)
  const [completedSteps, setCompletedSteps] = useState([])

  const showToast = (message, type = 'info') => {
    setToast({ message, type })
  }

  // NEW: Step completion handler
  const completeStep = (step) => {
    if (!completedSteps.includes(step)) {
      setCompletedSteps([...completedSteps, step])
    }
  }

  // Handler to view extracted features
  const handleViewFeatures = async () => {
    if (!imageId || selectedObjects.length === 0) {
      showToast('Please select an object first', 'warning')
      return
    }
    
    const objectId = selectedObjects[0]
    
    try {
      const result = await api.getFeatures(imageId, objectId)
      setCurrentFeatures(result.features)
      setShowFeatures(true)
    } catch (error) {
      try {
        await api.extractFeatures(imageId, objectId)
        const result = await api.getFeatures(imageId, objectId)
        setCurrentFeatures(result.features)
        setShowFeatures(true)
      } catch (extractError) {
        showToast('Failed to load features: ' + extractError.message, 'error')
      }
    }
  }

  const handleImageUpload = async (file) => {
    setUploadedImage(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target.result)
    }
    reader.readAsDataURL(file)
    
    await uploadAndDetect(file)
  }

  const uploadAndDetect = async (file) => {
    setIsProcessing(true)
    setSearchResults([])
    setCompletedSteps([]) // Reset workflow
    
    try {
      // 1. Upload image
      const uploadResult = await api.uploadImage(file)
      const newImageId = uploadResult.uploaded[0].image_id
      setImageId(newImageId)
      
      // 2. Detect objects
      const detectResult = await api.detectObjects(newImageId)
      
      // 3. Format detections
      const formattedDetections = detectResult.detections.map((det, idx) => ({
        id: idx,
        label: det.class,
        class: det.class,
        confidence: det.confidence,
        bbox: det.bbox
      }))
      
      setDetectedObjects(formattedDetections)
      setSelectedObjects(formattedDetections.length > 0 ? [0] : [])
      
      // 4. Complete step 1 and move to step 2
      if (formattedDetections.length > 0) {
        console.log('✅ Detection complete, imageId:', newImageId)
        showToast(`Detected ${formattedDetections.length} object(s)`, 'success')
        completeStep(1)
        
        // Wait a tiny bit to ensure state is updated
        setTimeout(() => {
          setCurrentStep(2) // Auto-advance to detection step
        }, 100)
        
        // Pre-extract features in background
        api.extractFeatures(newImageId, 0).catch(err => 
          console.error('Background feature extraction failed:', err)
        )
      }
          
    } catch (error) {
      console.error('Error:', error)
      showToast('Failed to process image: ' + error.message, 'error')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClearImage = () => {
    setUploadedImage(null)
    setImagePreview(null)
    setDetectedObjects([])
    setSelectedObjects([])
    setSearchResults([])
    setImageId(null)
    setCurrentStep(1)
    setCompletedSteps([])
  }

  const toggleObjectSelection = (id) => {
    setSelectedObjects(prev => 
      prev.includes(id) 
        ? prev.filter(objId => objId !== id)
        : [...prev, id]
    )
  }

  const handleWeightsChange = (weights) => {
    setFeatureWeights(weights)
  }

  const handleSearch = async (weights) => {
  if (selectedObjects.length === 0) {
    showToast('Please select at least one detected object to search', 'warning')
    return
  }

  setIsSearching(true)
  
  try {
    const objectId = selectedObjects[0]
    
    // Extract features first (ensure they exist)
    console.log('🔧 Ensuring features are extracted...', { imageId, objectId })
    await api.extractFeatures(imageId, objectId)
    
    // ✅ FIX: Handle null weights (automatic mode)
    let apiWeights = null
    if (weights) {
      // Manual weights from Advanced Mode
      apiWeights = {
        color: weights.color / 100,
        texture_tamura: weights.texture / 300,
        texture_gabor: weights.texture / 300,
        texture_lbp: weights.texture / 300,
        shape_hu: weights.shape / 300,
        shape_hog: weights.shape / 300,
        shape_contour: weights.shape / 300
      }
    }
    // If weights is null, backend will use automatic class-specific weights
    
    console.log('🔍 Searching with weights:', apiWeights ? 'custom' : 'automatic')
    
    // Search
    const searchResult = await api.searchSimilar(imageId, objectId, 20, apiWeights)
    
    console.log('✅ Search results:', searchResult.similar_objects.length)
    
    // Format results
    const formattedResults = searchResult.similar_objects.map((obj) => ({
      id: `${obj.image_id}-${obj.object_id}`,
      imageId: obj.image_id,
      objectId: obj.object_id,
      filename: obj.filename,
      similarity: Math.round(Math.min(obj.similarity * 100, 100)),
      className: obj.class,
      confidence: Math.round(obj.confidence * 100),
      bbox: obj.bbox
    }))
    
    setSearchResults(formattedResults)
    
    if (formattedResults.length > 0) {
      showToast(`Found ${formattedResults.length} similar images`, 'success')
      completeStep(3)
      setCurrentStep(4) // Move to results
    } else {
      showToast('No similar images found', 'info')
    }
    
  } catch (error) {
    console.error('Search error:', error)
    showToast('Search failed: ' + error.message, 'error')
  } finally {
    setIsSearching(false)
  }
}
  const handleUseImageAsQuery = async (imageId, filename) => {
    setCurrentPage('upload')
    setCurrentStep(1)
    setCompletedSteps([])
    
    const imageUrl = api.getImageUrl(filename)
    setImagePreview(imageUrl)
    setImageId(imageId)
    
    setIsProcessing(true)
    try {
      const detectResult = await api.detectObjects(imageId)
      
      const formattedDetections = detectResult.detections.map((det, idx) => ({
        id: idx,
        label: det.class,
        class: det.class,
        confidence: det.confidence,
        bbox: det.bbox
      }))
      
      setDetectedObjects(formattedDetections)
      setSelectedObjects(formattedDetections.length > 0 ? [0] : [])
      
      if (formattedDetections.length > 0) {
        showToast('Image loaded as query', 'success')
        completeStep(1)
        setCurrentStep(2)
        
        api.extractFeatures(imageId, 0).catch(err => 
          console.error('Background feature extraction failed:', err)
        )
      }
    } catch (error) {
      console.error('Error:', error)
      showToast('Failed to process image: ' + error.message, 'error')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
  <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
    {/* Header - Hidden on upload page, shown on gallery */}
    {currentPage === 'gallery' && (
      <Header currentPage={currentPage} onNavigate={setCurrentPage} />
    )}

    {/* Workflow Navigation - Only on upload page */}
    {currentPage === 'upload' && (
      <div className="fixed top-0 left-0 right-0 z-50 bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gradient">SmartGallery CBIR</h1>
          <button
            onClick={() => setCurrentPage('gallery')}
            className="btn-secondary text-sm"
          >
            View Gallery
          </button>
        </div>
        <WorkflowTabs 
          currentStep={currentStep}
          completedSteps={completedSteps}
          onStepClick={setCurrentStep}
        />
      </div>
    )}

    {/* Toast Notification */}
    {toast && (
      <Toast
        message={toast.message}
        type={toast.type}
        onClose={() => setToast(null)}
      />
    )}

    {currentPage === 'upload' ? (
      <>
        {/* Main Content with top padding for fixed header */}
        <main className="pt-[160px] px-6 pb-6">
          {/* Step 1: Upload */}
          {currentStep === 1 && (
            <UploadStep
              imagePreview={imagePreview}
              onImageUpload={handleImageUpload}
              onNavigateToGallery={() => setCurrentPage('gallery')}
              isProcessing={isProcessing}
            />
          )}

          {/* Step 2: Object Detection & Selection */}
          {currentStep === 2 && (
            <DetectionStep
              imagePreview={imagePreview}
              detectedObjects={detectedObjects}
              selectedObjects={selectedObjects}
              onToggleSelection={toggleObjectSelection}
              onClear={handleClearImage}
              onNext={() => {
                completeStep(2)
                setCurrentStep(3)
              }}
              onViewFeatures={handleViewFeatures}
              imageId={imageId}  // ← This line exists but check if it's there
            />
          )}

          {/* Step 3: Search Configuration */}
          {currentStep === 3 && (
            <SearchConfigStep
              featureWeights={featureWeights}
              onWeightsChange={handleWeightsChange}
              onSearch={(weights) => handleSearch(weights)}
              selectedObject={detectedObjects[selectedObjects[0]]}
              imageId={imageId}
            />
          )}

          {/* Step 4: Results */}
          {currentStep === 4 && (
            <ResultsStep
              results={searchResults}
              isSearching={isSearching}
            />
          )}
        </main>
      </>
    ) : (
      <main className="pt-20 px-6 pb-6">
        <Gallery onUseAsQuery={handleUseImageAsQuery} showToast={showToast} />
      </main>
    )}

    {/* Feature Viewer Modal */}
    {showFeatures && (
      <FeatureViewer
        features={currentFeatures}
        onClose={() => setShowFeatures(false)}
      />
    )}
  </div>
)
}
export default App