import React, { useState } from 'react'

function DocumentUpload({ onUpload, loading }) {
  const [dragActive, setDragActive] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = async (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      await handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file) => {
    if (!file.name.endsWith('.pdf')) {
      setUploadStatus({ success: false, message: 'Only PDF files are supported' })
      return
    }

    setUploadStatus({ success: null, message: 'Uploading...' })
    const result = await onUpload(file)
    setUploadStatus(result)

    // Clear status after 3 seconds
    setTimeout(() => setUploadStatus(null), 3000)
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        📄 Upload Document
      </h2>
      
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".pdf"
          onChange={handleChange}
          disabled={loading}
        />
        
        <label
          htmlFor="file-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          <svg
            className="w-16 h-16 text-gray-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <p className="text-gray-600 mb-2">
            Drag and drop a PDF file here, or click to select
          </p>
          <p className="text-sm text-gray-500">Only PDF files are supported</p>
        </label>
      </div>

      {uploadStatus && (
        <div
          className={`mt-4 p-3 rounded ${
            uploadStatus.success
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}
        >
          <div className="flex items-center gap-2">
            {uploadStatus.success && uploadStatus.filename && (
              <span className="font-semibold">📎 {uploadStatus.filename}</span>
            )}
            <span>{uploadStatus.message}</span>
          </div>
        </div>
      )}

      {loading && (
        <div className="mt-4 text-center">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
          <p className="text-sm text-gray-600 mt-2">Processing document...</p>
        </div>
      )}
    </div>
  )
}

export default DocumentUpload
