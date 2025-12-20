// frontend/src/components/WorkflowTabs.jsx
import { CheckCircle, Upload, Target, Settings, Grid } from 'lucide-react'

const steps = [
  { id: 1, name: 'Upload', icon: Upload },
  { id: 2, name: 'Detect Objects', icon: Target },
  { id: 3, name: 'Configure Search', icon: Settings },
  { id: 4, name: 'Results', icon: Grid },
]

export default function WorkflowTabs({ currentStep, onStepClick, completedSteps }) {
  return (
    <div className="sticky top-16 bg-white border-b border-gray-200 shadow-sm z-40">
      <div className="max-w-7xl mx-auto px-6">
        <nav className="flex -mb-px">
          {steps.map((step, idx) => {
            const isActive = currentStep === step.id
            const isCompleted = completedSteps.includes(step.id)
            const isClickable = isCompleted || idx === 0
            const Icon = step.icon

            return (
              <button
                key={step.id}
                onClick={() => isClickable && onStepClick(step.id)}
                disabled={!isClickable}
                className={`
                  group relative flex-1 py-4 px-1 text-center border-b-2 font-medium text-sm
                  transition-all duration-200
                  ${isActive 
                    ? 'border-blue-500 text-blue-600' 
                    : isCompleted 
                      ? 'border-green-500 text-green-600 hover:border-green-600' 
                      : 'border-transparent text-gray-400 cursor-not-allowed'
                  }
                  ${isClickable && !isActive ? 'hover:text-gray-600 hover:border-gray-300' : ''}
                `}
              >
                <div className="flex items-center justify-center space-x-2">
                  <div className={`
                    flex items-center justify-center w-8 h-8 rounded-full
                    ${isActive 
                      ? 'bg-blue-100 text-blue-600' 
                      : isCompleted 
                        ? 'bg-green-100 text-green-600' 
                        : 'bg-gray-100 text-gray-400'
                    }
                  `}>
                    {isCompleted && !isActive ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                  <span className="hidden sm:block">{step.name}</span>
                </div>
              </button>
            )
          })}
        </nav>
      </div>
    </div>
  )
}