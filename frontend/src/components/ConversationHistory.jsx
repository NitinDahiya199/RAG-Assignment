import React, { useState } from 'react'

function ConversationHistory({ conversations }) {
  const [expanded, setExpanded] = useState(false)

  const allConversations = Object.entries(conversations).flatMap(([userId, convs]) =>
    convs.map(conv => ({ ...conv, userId }))
  )

  if (allConversations.length === 0) {
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-gray-800">
          💭 Conversation History
        </h2>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-sm text-primary-600 hover:text-primary-700"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>

      {expanded && (
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {allConversations.map((conv, idx) => (
            <div key={idx} className="border-l-4 border-primary-500 pl-4 py-2">
              <div className="mb-2">
                <p className="font-semibold text-gray-800">Q: {conv.question}</p>
                <p className="text-sm text-gray-500 mt-1">
                  {new Date(conv.timestamp).toLocaleString()}
                </p>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <p className="text-gray-700">{conv.answer}</p>
                {conv.confidence !== undefined && (
                  <p className="text-xs text-gray-500 mt-2">
                    Confidence: {(conv.confidence * 100).toFixed(1)}%
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ConversationHistory
