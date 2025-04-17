'use client'

import useFetch from '@/utils/useFetch'
import React, { useEffect, useState } from 'react'
import TranscriptItem from '@/components/TranscriptItem' // Import the new component
import { Status } from '../page'
import { Button } from '@heroui/button'

// Define a type for the data items
interface TranscriptData {
    id: number
    speaker: string
    start_time: number
    end_time: number
    duration: number
    file_path: string
    text: string
    translated_text: string
}

const Page = () => {
    const { fetchData } = useFetch()
    const [data, setData] = useState<TranscriptData[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const [loading, setLoading] = useState<boolean>(false)
    const [uploadStatus, setUploadStatus] = useState<Status | null>(null)

    useEffect(() => {
        const getData = async () => {
            setIsLoading(true)
            setError(null)
            try {
                const result = await fetchData<TranscriptData[]>('/phase2/final-result', {
                    method: 'GET',
                })

                if (result) {
                    // Sort the json object by start_time
                    const sortedData = result.sort((a, b) => a.start_time - b.start_time)
                    setData(sortedData)
                } else {
                    setError('Failed to fetch data or data is empty.')
                }
            } catch (err) {
                console.error('Error fetching data:', err)
                setError('An error occurred while fetching data.')
            } finally {
                setIsLoading(false)
            }
        }
        getData()
    }, []) // Add fetchData as dependency

    const handleTextChange = (id: number, newText: string) => {
        setData((prevData) =>
            prevData.map((item) => (item.id === id ? { ...item, translated_text: newText } : item))
        )
    }

    const handleSave = async () => {
        setLoading(true)
        const response = await fetchData<any>('/phase3/training', {
            method: 'POST',
            body: JSON.stringify({ data: data }),
        })

        if (response) {
            setUploadStatus({
                mode: 300,
                message: response.message,
            })

            // Create a timeout loop to call the backend to get the latest details
            const interval = setInterval(async () => {
                const response = await fetchData<any>('/status/phase3', {
                    method: 'GET',
                })

                if (response) {
                    if (response.is_complete === true) {
                        clearInterval(interval)
                        setLoading(false)

                        window.location.href = '/phase4'
                    } else {
                        setUploadStatus({ mode: 300, message: response.message })
                    }
                } else {
                    clearInterval(interval)
                    setLoading(false)
                }
            }, 2000)
        } else {
            setUploadStatus({ mode: 500, message: 'Save failed' })
            setLoading(false)
        }
    }

    const testGen = async () => {
        const response = await fetchData<any>('/phase3/generate', {
            method: 'GET',
        })

        console.log('Test gen:', response)
    }

    return (
        <div className="container mx-auto p-4 dark:bg-gray-900 dark:text-white">
            <div className="mb-4 flex items-center justify-between">
                <h1 className="text-2xl font-bold">Transcript Editor</h1>
                <button
                    onClick={handleSave}
                    className="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-800">
                    Save Changes
                </button>
                <Button onPress={testGen}>Test Gen</Button>
            </div>

            {isLoading && <p>Loading transcript...</p>}
            {error && <p className="text-red-500">{error}</p>}

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

            {!isLoading && !error && (
                <div>
                    {data.map((item) => (
                        <TranscriptItem key={item.id} item={item} onTextChange={handleTextChange} />
                    ))}
                </div>
            )}
            {/* Optional: Display raw data for debugging */}
            {/* <div className="mt-8 rounded bg-gray-100 p-4 dark:bg-gray-800">
                <h2 className="mb-2 text-lg font-semibold">Raw Data</h2>
                <pre className="overflow-x-auto text-xs">
                    <code>{JSON.stringify(data, null, 2)}</code>
                </pre>
            </div> */}
        </div>
    )
}

export default Page
