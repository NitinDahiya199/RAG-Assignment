import React from 'react'

function StatsPanel({ stats, documents, documentsCount, onClear }) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-gray-800">📊 Statistics</h2>
        {documentsCount > 0 && (
          <button
            onClick={onClear}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
          >
            Clear All Documents
          </button>
        )}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Documents Processed</p>
          <p className="text-3xl font-bold text-blue-600">{documentsCount}</p>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Total Chunks</p>
          <p className="text-3xl font-bold text-green-600">
            {stats?.total_documents || 0}
          </p>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Vector Store</p>
          <p className="text-lg font-semibold text-purple-600">
            {stats ? 'Active' : 'Empty'}
          </p>
        </div>
      </div>

      {/* Uploaded Files List */}
      {documents && documents.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">📄 Uploaded Documents</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {documents.map((doc, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate" title={doc.filename}>
                    📎 {doc.filename}
                  </p>
                  {doc.title && doc.title !== doc.filename && (
                    <p className="text-xs text-gray-500 truncate mt-1">{doc.title}</p>
                  )}
                  {doc.chunks > 0 && (
                    <p className="text-xs text-gray-400 mt-1">{doc.chunks} chunks</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default StatsPanel
