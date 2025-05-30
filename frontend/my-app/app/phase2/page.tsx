'use client'

import useFetch, { baseUrl } from '@/utils/useFetch'
import { Button } from '@heroui/button'
import React, { use, useEffect, useState } from 'react'
import AudioCard from '@/components/phase1_audio_card'
import { Status } from '../page'
import Swal from 'sweetalert2'

const page = () => {
    const { fetchData } = useFetch()
    const [data, setData] = useState<any[]>([])
    const [speakers, setSpeakers] = useState<string[]>([])
    const [selectedAudioIds, setSelectedAudioIds] = useState<number[]>([])
    const [loading, setLoading] = useState<boolean>(false)
    const [uploadStatus, setUploadStatus] = useState<Status | null>(null)

    useEffect(() => {
        const getData = async () => {
            const result = await fetchData<any>('/phase2/result', {
                method: 'GET',
            })

            if (result) {
                // This is a list of objects, loop though it to get the speakers and set it to the speakers state
                const get_speakers: string[] = []
                for (const item of result) {
                    if (!get_speakers.includes(item.speaker)) {
                        get_speakers.push(item.speaker)
                    }
                }

                setSpeakers(get_speakers)

                // Sort the json object by start_time
                const sortedData = result.sort((a: any, b: any) => a.start_time - b.start_time)

                setData(sortedData)
            }
        }
        getData()
    }, [])

    const handleSaveAndContinue = async () => {
        // Implement save and continue logic here
        setLoading(true)
        const response = await fetchData<any>('/phase2/save', {
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
                const response = await fetchData<any>('/status/phase2', {
                    method: 'GET',
                })

                if (response) {
                    if (response.is_complete === true) {
                        clearInterval(interval)
                        setLoading(false)

                        window.location.href = '/phase3'
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

    const handleCombine = async () => {
        // Implement combine logic here
        if (selectedAudioIds.length < 2) {
            alert('Please select at least two audio segments to combine.')
            return
        }
        // alert(`Selected IDs for combination: ${selectedAudioIds.join(', ')}`)

        // Future combine implementation would go here
        setLoading(true)
        console.log({ combine: selectedAudioIds })
        const response = await fetchData<any>('/phase2/combine-audio', {
            method: 'POST',
            body: JSON.stringify({ combine: selectedAudioIds }),
        })

        if (response) {
            setUploadStatus({
                mode: 300,
                message: response.message,
            })

            // Create a timeout loop to call the backend to get the latest details
            const interval = setInterval(async () => {
                const response = await fetchData<any>('/status/combine', {
                    method: 'GET',
                })

                if (response) {
                    if (response.is_complete === true) {
                        clearInterval(interval)
                        setLoading(false)

                        handleRefresh()
                    } else {
                        setUploadStatus({ mode: 300, message: response.message })
                    }
                } else {
                    clearInterval(interval)
                    setLoading(false)
                }
            }, 2000)
        }
        setLoading(false)
    }

    const handleDelete = async () => {
        // Implement delete logic for selected audio segments
        if (selectedAudioIds.length === 0) {
            alert('Please select at least one audio segment to delete.')
            return
        }

        const response = await fetchData<any>('/phase2/delete-audio', {
            method: 'POST',
            body: JSON.stringify({ delete: selectedAudioIds }),
        })
        if (response) {
            handleRefresh()
        }
    }

    const handleRefresh = () => {
        window.location.reload()
    }

    const handleTextChange = (index: number, newText: string) => {
        const newData = [...data]
        newData[index].text = newText
        setData(newData)
    }

    const handleSpeakerChange = (index: number, newSpeaker: string) => {
        console.log('handleSpeakerChange', index, newSpeaker)
        const newData = [...data]
        newData[index].speaker = newSpeaker
        setData(newData)
    }

    const handleSelectionChange = (id: number, isSelected: boolean) => {
        if (isSelected) {
            // Add ID to selected IDs if not already there
            setSelectedAudioIds((prev) => [...prev.filter((prevId) => prevId !== id), id])
        } else {
            // Remove ID from selected IDs
            setSelectedAudioIds((prev) => prev.filter((prevId) => prevId !== id))
        }
    }

    return (
        <div className="container mx-auto px-4 py-6">
            <div className="mb-6">
                <div className="flex flex-wrap gap-3 mb-2">
                    <Button
                        onPress={handleCombine}
                        className="bg-blue-500 hover:bg-blue-600"
                        isDisabled={loading}>
                        Combine
                    </Button>
                    <Button
                        onPress={handleDelete}
                        className="bg-red-500 hover:bg-red-600"
                        isDisabled={loading}>
                        Delete
                    </Button>
                    <Button
                        onPress={handleRefresh}
                        className="bg-gray-500 hover:bg-gray-600"
                        isDisabled={loading}>
                        Refresh
                    </Button>
                    {selectedAudioIds.length > 0 && (
                        <div className="text-sm text-gray-600 dark:text-gray-300 mt-2">
                            Selected: {selectedAudioIds.length} item(s) (IDs:{' '}
                            {selectedAudioIds.join(', ')})
                        </div>
                    )}
                </div>

                <Button
                    onPress={handleSaveAndContinue}
                    isDisabled={loading}
                    className="mt-8 bg-green-500 hover:bg-green-600">
                    Save and Continue
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
            </div>

            <div className="space-y-4">
                {data.map((item, index) => (
                    <AudioCard
                        key={index}
                        aria-label={`Audio segment ${index + 1}`}
                        item={item}
                        index={index}
                        speakers={speakers}
                        onTextChange={handleTextChange}
                        onSpeakerChange={handleSpeakerChange}
                        onSelectionChange={handleSelectionChange}
                        isSelected={selectedAudioIds.includes(item.id)}
                    />
                ))}
            </div>
        </div>
    )
}

export default page
