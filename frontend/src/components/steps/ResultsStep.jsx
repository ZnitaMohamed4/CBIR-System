// frontend/src/components/steps/ResultsStep.jsx
import SearchResults from '../SearchResults'

export default function ResultsStep({ results, isSearching }) {
  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Search Results
        </h2>
        <p className="text-gray-600">
          {results.length > 0 
            ? `Found ${results.length} similar images` 
            : 'No results yet'}
        </p>
      </div>

      <SearchResults results={results} isLoading={isSearching} />
    </div>
  )
}