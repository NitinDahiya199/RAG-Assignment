import React, { useState, useEffect } from 'react'
import axios from 'axios'
import DocumentUpload from './components/DocumentUpload'
import QueryInterface from './components/QueryInterface'
import ConversationHistory from './components/ConversationHistory'
import StatsPanel from './components/StatsPanel'

// Use proxy in development, direct URL in production
const API_BASE_URL = import.meta.env.DEV 
  ? '/api'  // Use Vite proxy in development
  : 'http://localhost:8000'  // Direct URL in production

function App() {
  const [documents, setDocuments] = useState([]) // Array of document objects with filename
  const [documentsCount, setDocumentsCount] = useState(0)
  const [stats, setStats] = useState(null)
  const [conversations, setConversations] = useState({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`)
      setStats(response.data.stats)
      setDocumentsCount(response.data.documents_count || 0)
      // Keep existing documents list, only update count
    } catch (err) {
      console.error('Error fetching stats:', err)
    }
  }

  const handleDocumentUpload = async (file) => {
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      
      if (response.data.success) {
        // Add the uploaded document to the list
        const newDocument = {
          filename: response.data.document?.filename || file.name,
          title: response.data.document?.title || file.name,
          chunks: response.data.document?.chunks_count || response.data.chunks_added || 0,
          uploadedAt: new Date().toISOString()
        }
        setDocuments(prev => [...prev, newDocument])
        await fetchStats()
        return { 
          success: true, 
          message: response.data.message || `Successfully uploaded: ${file.name}`,
          filename: file.name
        }
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Error uploading document'
      setError(errorMsg)
      return { success: false, message: errorMsg }
    } finally {
      setLoading(false)
    }
  }

  const handleQuery = async (question, userId = 'default') => {
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('question', question)
      formData.append('user_id', userId)
      formData.append('top_k', '5')
      
      const response = await axios.post(`${API_BASE_URL}/query`, formData)
      
      if (response.data.success) {
        // Update conversation history
        if (!conversations[userId]) {
          conversations[userId] = []
        }
        conversations[userId].push({
          question,
          answer: response.data.answer,
          confidence: response.data.confidence,
          sources: response.data.sources,
          timestamp: new Date().toISOString()
        })
        setConversations({ ...conversations })
        
        return response.data
      }
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Error processing query'
      setError(errorMsg)
      throw new Error(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleClearDocuments = async () => {
    try {
      await axios.delete(`${API_BASE_URL}/documents`)
      await fetchStats()
      setDocuments([]) // Clear the documents list
      setDocumentsCount(0)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error clearing documents')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            📚 Document Q&A AI Agent
          </h1>
          <p className="text-gray-600">
            Enterprise-Ready AI Agent powered by Google Gemini API
          </p>
        </header>

        {/* Error Banner */}
        {error && (
          <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
            <span className="block sm:inline">{error}</span>
            <button
              className="absolute top-0 bottom-0 right-0 px-4 py-3"
              onClick={() => setError(null)}
            >
              ×
            </button>
          </div>
        )}

        {/* Stats Panel */}
        <StatsPanel 
          stats={stats} 
          documents={documents} 
          documentsCount={documentsCount}
          onClear={handleClearDocuments} 
        />

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Left Column - Document Upload */}
          <div className="lg:col-span-1">
            <DocumentUpload
              onUpload={handleDocumentUpload}
              loading={loading}
            />
          </div>

          {/* Right Column - Query Interface */}
          <div className="lg:col-span-2">
            <QueryInterface
              onQuery={handleQuery}
              loading={loading}
            />
          </div>
        </div>

        {/* Conversation History */}
        <ConversationHistory conversations={conversations} />
      </div>
    </div>
  )
}

export default App
