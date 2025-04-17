'use client'

import React, { useEffect, useState } from 'react'
import { Button } from '@heroui/button'
import { Input } from '@heroui/input'
import useFetch from '@/utils/useFetch'
import { useRouter } from 'next/navigation'

export type Status = {
    message: string
    mode: number
}

const languageOptions = [
    { value: 'Cantonese', label: 'Cantonese' },
    { value: 'Mandarin', label: 'Mandarin' },
    { value: 'English', label: 'English' },
    { value: 'Korean', label: 'Korean' },
    { value: 'Japanese', label: 'Japanese' },
]

export default function Home() {
    const [videoFile, setVideoFile] = useState<File | null>(null)
    const [uploadStatus, setUploadStatus] = useState<Status | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [loading, setLoading] = useState<boolean>(false)
    const [inputLanguage, setInputLanguage] = useState<string>('English')
    const [outputLanguage, setOutputLanguage] = useState<string>('Cantonese')
    const { fetchData, loading: uploading } = useFetch()

    const router = useRouter()

    useEffect(() => {
        setLoading(uploading)
    }, [loading])

    // Handle file selection
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null
        setVideoFile(file)

        // Create preview URL if file exists
        if (file) {
            const objectUrl = URL.createObjectURL(file)
            setPreview(objectUrl)
            setUploadStatus(null)
        } else {
            setPreview(null)
        }
    }

    // Handle form submission
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!videoFile) {
            setUploadStatus({ mode: 500, message: 'Please select a video file' })
            return
        }

        setUploadStatus({ mode: 300, message: 'Uploading ...' })
        setLoading(true)

        const formData = new FormData()
        formData.append('file', videoFile)
        formData.append('input_language', inputLanguage)
        formData.append('output_language', outputLanguage)

        const result = await fetchData<{ message: string }>('/upload-video', {
            method: 'POST',
            body: formData,
        })

        if (result) {
            setUploadStatus({ mode: 300, message: result.message })

            // Create a timeout loop to call the backend to get the latest details
            const interval = setInterval(async () => {
                const response = await fetchData<any>('/status/phase1', {
                    method: 'GET',
                })

                if (response) {
                    if (response.is_complete === true) {
                        clearInterval(interval)
                        setLoading(false)
                        // Router push to the next page
                        router.push('/phase2')
                    } else {
                        setUploadStatus({ mode: 300, message: response.message })
                    }
                } else {
                    clearInterval(interval)
                    setLoading(false)
                }
            }, 2000)
        } else {
            // Error is already handled by useFetch with SweetAlert2
            setUploadStatus({ mode: 500, message: 'Upload failed' })
            setLoading(false)
        }
    }

    return (
        <div className="max-w-md mx-auto p-6">
            <h1 className="text-2xl font-bold mb-6">Video Upload</h1>

            <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                    <label htmlFor="video-upload" className="block text-sm font-medium">
                        Select Video
                    </label>
                    <Input
                        id="video-upload"
                        type="file"
                        accept="video/*"
                        onChange={handleFileChange}
                        className="w-full"
                    />
                    <p className="text-xs text-gray-500">
                        Select a video file to upload (.mp4, .mov, etc.)
                    </p>
                </div>

                {preview && (
                    <div className="mt-4">
                        <h2 className="text-sm font-medium mb-2">Preview:</h2>
                        <video
                            src={preview}
                            controls
                            className="w-full border rounded"
                            style={{ maxHeight: '300px' }}
                        />
                    </div>
                )}

                <div className="space-y-2">
                    <label htmlFor="input-language" className="block text-sm font-medium">
                        Input Language
                    </label>
                    <select
                        id="input-language"
                        value={inputLanguage}
                        onChange={(e) => setInputLanguage(e.target.value)}
                        className="w-full p-2 border rounded bg-white text-gray-900">
                        {languageOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="space-y-2">
                    <label htmlFor="output-language" className="block text-sm font-medium">
                        Output Language
                    </label>
                    <select
                        id="output-language"
                        value={outputLanguage}
                        onChange={(e) => setOutputLanguage(e.target.value)}
                        className="w-full p-2 border rounded bg-white text-gray-900">
                        {languageOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                </div>

                <Button
                    type="submit"
                    color="primary"
                    isLoading={uploading}
                    isDisabled={loading}
                    className="w-full">
                    {uploading ? 'Uploading...' : 'Upload Video'}
                </Button>

                {uploadStatus && (
                    <div
                        className={`mt-4 p-3 rounded text-sm ${
                            uploadStatus.mode === 500
                                ? 'bg-red-100 text-red-800'
                                : uploadStatus.mode === 300
                                  ? 'bg-yellow-100 text-yellow-800'
                                  : uploadStatus.mode === 200
                                    ? 'bg-green-100 text-green-800'
                                    : ''
                        }`}>
                        {uploadStatus.message}

                        {uploadStatus.mode === 300 && (
                            <div className="flex justify-center mt-2">
                                <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-green-500"></div>
                            </div>
                        )}
                    </div>
                )}
            </form>
        </div>
    )
}
