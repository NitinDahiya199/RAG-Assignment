import React, { useState } from 'react'

function QueryInterface({ onQuery, loading }) {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState(null)
  const [queryHistory, setQueryHistory] = useState([])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim() || loading) return

    try {
      const result = await onQuery(question)
      setResponse(result)
      setQueryHistory([...queryHistory, { question, response: result }])
      setQuestion('')
    } catch (error) {
      setResponse({
        success: false,
        answer: error.message || 'Error processing query',
        confidence: 0
      })
    }
  }

  const exampleQueries = [
    "What is the main contribution of this paper?",
    "Summarize the methodology",
    "What are the evaluation metrics?",
    "What is the conclusion?"
  ]

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        💬 Ask Questions
      </h2>

      <form onSubmit={handleSubmit} className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the documents..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={!question.trim() || loading}
            className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Processing...' : 'Ask'}
          </button>
        </div>
      </form>

      {/* Example Queries */}
      <div className="mb-4">
        <p className="text-sm text-gray-600 mb-2">Example queries:</p>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((example, idx) => (
            <button
              key={idx}
              onClick={() => setQuestion(example)}
              className="text-xs px-3 py-1 bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Response */}
      {response && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-gray-800">Answer</h3>
            {response.confidence !== undefined && (
              <span className="text-xs px-2 py-1 bg-primary-100 text-primary-800 rounded">
                Confidence: {(response.confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <p className="text-gray-700 whitespace-pre-wrap">{response.answer}</p>

          {/* Sources */}
          {response.sources && response.sources.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h4 className="text-sm font-semibold text-gray-600 mb-2">Sources:</h4>
              <ul className="space-y-1">
                {response.sources.map((source, idx) => (
                  <li key={idx} className="text-xs text-gray-600">
                    • {source.document || source.document_id} 
                    {source.page && ` (Page ${source.page})`}
                    {source.section && ` - ${source.section}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Loading Indicator */}
      {loading && (
        <div className="mt-4 text-center">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
          <p className="text-sm text-gray-600 mt-2">Processing your question...</p>
        </div>
      )}
    </div>
  )
}

export default QueryInterface
