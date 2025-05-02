'use client'

import useFetch from '@/utils/useFetch'
import React, { useEffect, useState } from 'react'
import { Status } from '../page'
import { baseUrl } from '@/utils/useFetch'

type Props = {}

// Define a more specific type for speakers
interface Speaker {
    name: string
}

const page = (props: Props) => {
    const { fetchData } = useFetch()

    const [imageGroup, setImageGroup] = useState<Record<string, string[]>>({})
    const [speakerList, setSpeakerList] = useState<Speaker[]>([]) // Updated type
    const [speakerAudio, setSpeakerAudio] = useState<Record<string, string>>({}) // New state for audio paths

    const [loading, setLoading] = useState<boolean>(false)
    const [uploadStatus, setUploadStatus] = useState<Status | null>(null)

    useEffect(() => {
        const getData = async () => {
            setLoading(true)
            const response = await fetchData<any>('/phase5/start_face_detection', {
                method: 'GET',
            })

            if (response) {
                setUploadStatus({
                    mode: 300,
                    message: response.message,
                })

                // Create a timeout loop to call the backend to get the latest details
                const interval = setInterval(async () => {
                    const response = await fetchData<any>('/status/phase5', {
                        method: 'GET',
                    })

                    if (response) {
                        if (response.is_complete === true) {
                            clearInterval(interval)
                            // setLoading(false) // Keep loading until final data is fetched

                            const finalResponse = await fetchData<any>(
                                '/phase5/get_face_detection',
                                {
                                    method: 'GET',
                                }
                            )

                            if (finalResponse) {
                                console.log(finalResponse)
                                setImageGroup(finalResponse.image_group)
                                setSpeakerList(finalResponse.speakers) // Update speaker list
                                setSpeakerAudio(finalResponse.speaker_audio) // Set speaker audio paths
                                setLoading(false) // Set loading false after data is set
                                setUploadStatus(null)
                            } else {
                                // Handle error fetching final data
                                setUploadStatus({
                                    mode: 500,
                                    message: 'Failed to fetch face detection results.',
                                })
                                setLoading(false)
                            }
                        } else {
                            setUploadStatus({ mode: 300, message: response.message })
                        }
                    } else {
                        clearInterval(interval)
                        setLoading(false)
                        setUploadStatus({ mode: 500, message: 'Failed to get status update.' }) // Inform user about status check failure
                    }
                }, 2000)
            } else {
                setUploadStatus({ mode: 500, message: 'Failed to start face detection.' }) // More specific error
                setLoading(false)
            }
        }
        getData()
        // Cleanup interval on component unmount
        // return () => clearInterval(interval); // This line causes issues because interval is defined inside the async function scope. Need refactoring if cleanup is strictly needed.
    }, [])

    const handleContinue = async () => {
        // Handle continue button click
        const selectedSpeakers = Object.fromEntries(
            Object.entries(imageGroup).map(([groupKey, images]) => {
                const selectEl = document.getElementById(
                    `speaker-select-${groupKey}`
                ) as HTMLSelectElement
                return [selectEl.value, parseInt(groupKey, 10)] as [string, number]
            })
        )

        console.log('Selected Speakers:', selectedSpeakers)

        setLoading(true)
        const response = await fetchData<any>('/phase5/generate_final_video', {
            method: 'POST',
            body: JSON.stringify({ data: selectedSpeakers }),
        })

        if (response) {
            setUploadStatus({
                mode: 300,
                message: response.message,
            })

            // Create a timeout loop to call the backend to get the latest details
            const interval = setInterval(async () => {
                const response = await fetchData<any>('/status/phase6', {
                    method: 'GET',
                })

                if (response) {
                    if (response.is_complete === true) {
                        clearInterval(interval)
                        setLoading(false)
                        setUploadStatus(null)

                        // TODO
                    } else {
                        setUploadStatus({ mode: 300, message: response.message })
                    }
                } else {
                    clearInterval(interval)
                    setLoading(false)
                    setUploadStatus({ mode: 500, message: 'Failed to get status update.' }) // Inform user about status check failure
                }
            }, 2000)
        } else {
            setUploadStatus({ mode: 500, message: 'Failed to start face detection.' }) // More specific error
            setLoading(false)
        }
    }

    return (
        <div className="container mx-auto px-4 py-6 text-gray-900 dark:text-gray-100">
            {/* Status Message */}
            {uploadStatus && (
                <div
                    className={`mb-4 p-3 rounded text-sm ${
                        uploadStatus.mode === 500
                            ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-100'
                            : uploadStatus.mode === 300
                              ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-100'
                              : uploadStatus.mode === 200
                                ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-100'
                                : ''
                    }`}>
                    {uploadStatus.message}
                    {/* Loading Spinner */}
                    {(loading || uploadStatus.mode === 300) && ( // Show spinner if loading state is true OR status is 300
                        <div className="flex justify-center mt-2">
                            <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-blue-500 dark:border-blue-300"></div>
                        </div>
                    )}
                </div>
            )}

            {/* Speaker Audio Preview Section */}
            {!loading && speakerList.length > 0 && Object.keys(speakerAudio).length > 0 && (
                <div className="mb-6 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg shadow">
                    <h2 className="text-xl font-semibold mb-3 text-center">
                        Speaker Audio Preview
                    </h2>
                    <div className="space-y-3">
                        {speakerList.map((speaker) => {
                            const audioPath = speakerAudio[speaker.name]
                            if (!audioPath) return null // Skip if no audio path found for the speaker
                            const audioUrl = `${baseUrl}/public/phase1/${audioPath.replace(/\\/g, '/')}` // Construct URL and replace backslashes
                            return (
                                <div
                                    key={speaker.name}
                                    className="flex items-center justify-between p-2 bg-white dark:bg-gray-800 rounded shadow-sm">
                                    <span className="font-medium">{speaker.name}</span>
                                    <audio controls src={audioUrl} className="h-8">
                                        Your browser does not support the audio element.
                                    </audio>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Image Groups Display */}
            {!loading && Object.keys(imageGroup).length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(imageGroup).map(([groupKey, images], index) => (
                        <div
                            key={groupKey}
                            className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
                            <h3 className="text-lg font-semibold mb-2 text-center">
                                Group {parseInt(groupKey) + 1}
                            </h3>
                            <div className="flex justify-around items-center mb-4 space-x-2">
                                {images.map((imageName) => (
                                    <img
                                        key={imageName}
                                        src={`${baseUrl}/public/phase5/images/${imageName}`}
                                        alt={`Face ${imageName}`}
                                        className="w-24 h-24 object-cover rounded border border-gray-300 dark:border-gray-600"
                                    />
                                ))}
                            </div>
                            <div className="mt-4">
                                <label
                                    htmlFor={`speaker-select-${groupKey}`}
                                    className="block text-sm font-medium mb-1">
                                    Assign Speaker:
                                </label>
                                <select
                                    id={`speaker-select-${groupKey}`}
                                    name={`speaker-select-${groupKey}`}
                                    defaultValue={speakerList[index]?.name || ''} // Default to speaker name at the corresponding index
                                    className="w-full p-2 border border-gray-300 rounded-md bg-white dark:bg-gray-700 dark:border-gray-600 focus:ring-blue-500 focus:border-blue-500">
                                    <option value="" disabled>
                                        Select Speaker
                                    </option>
                                    {/* Map over speakerList using speaker objects */}
                                    {speakerList.map((speaker) => (
                                        <option key={speaker.name} value={speaker.name}>
                                            {speaker.name}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Continue Button */}
            {!loading && Object.keys(imageGroup).length > 0 && (
                <div className="mt-6 flex justify-end">
                    <button
                        onClick={handleContinue}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md shadow focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600">
                        Continue
                    </button>
                </div>
            )}
        </div>
    )
}

export default page
