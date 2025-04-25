'use client'

import React, { useState, useEffect } from 'react'
import { Slider } from '@heroui/slider'
import useFetch from '@/utils/useFetch'
import Swal from 'sweetalert2'

interface AudioSegment {
    id: number
    speaker: string
    start_time: number
    end_time: number
    duration: number
    file_path: string
    text: string
    translated_text: string
    generated_audio_duration: number
    generated_audio_speed: number
}

interface AudioDisplayProps {
    data: AudioSegment[]
    originalAudioBaseUrl: string
    generatedAudioBaseUrl: string
    selectedModels: any
}

const AudioDisplay: React.FC<AudioDisplayProps> = ({
    data,
    originalAudioBaseUrl,
    generatedAudioBaseUrl,
    selectedModels,
}) => {
    const [speeds, setSpeeds] = useState<{ [key: number]: number }>({})
    const [editedTexts, setEditedTexts] = useState<{ [key: number]: string }>({})

    const { fetchData } = useFetch()

    useEffect(() => {
        const initialSpeeds = data.reduce(
            (acc, segment) => {
                acc[segment.id] = segment.generated_audio_speed
                return acc
            },
            {} as { [key: number]: number }
        )
        const initialTexts = data.reduce(
            (acc, segment) => {
                acc[segment.id] = segment.translated_text
                return acc
            },
            {} as { [key: number]: string }
        )
        setSpeeds(initialSpeeds)
        setEditedTexts(initialTexts)
    }, [data])

    const handleSpeedChange = (id: number, value: number | number[]) => {
        const newSpeed = Array.isArray(value) ? value[0] : value
        setSpeeds((prevSpeeds) => ({
            ...prevSpeeds,
            [id]: newSpeed,
        }))
    }

    const handleTextChange = (id: number, newText: string) => {
        setEditedTexts((prevTexts) => ({
            ...prevTexts,
            [id]: newText,
        }))
    }

    const handleRegenerate = async (id: number) => {
        const segmentData = {
            id: id,
            speed: speeds[id],
            translated_text: editedTexts[id],
            selectedModels: selectedModels,
        }
        console.log('Re-generate clicked:', JSON.stringify(segmentData, null, 2))

        /*
            id: int = None,
            speed: float = None,
            translated_text: str = None,
            selectedModels: dict = None,
         */
        const response = await fetchData<any>('/phase4/re-generate', {
            method: 'POST',
            body: JSON.stringify(segmentData),
        })

        Swal.showLoading()

        if (response) {
            const interval = setInterval(async () => {
                try {
                    const statusResponse = await fetchData<any>('/status/phase4', {
                        method: 'GET',
                    })

                    if (statusResponse) {
                        if (statusResponse.is_complete === true) {
                            clearInterval(interval)
                            window.location.reload()
                        }
                    } else {
                        clearInterval(interval)
                        Swal.fire({
                            icon: 'error',
                            title: 'Error',
                            text: 'Failed to fetch status.',
                        })
                    }
                } catch (error) {
                    console.error('Error fetching status:', error)
                    clearInterval(interval)
                    Swal.fire({
                        icon: 'error',
                        title: 'Error',
                        text: 'Failed to fetch status.',
                    })
                }
            }, 3000)
        } else {
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: 'Failed to re-generate audio.' + response,
            })
        }
    }

    const formatTime = (time: number) => time.toFixed(2)

    if (!data || data.length === 0) {
        return <p className="dark:text-gray-400">No audio data to display.</p>
    }

    return (
        <div className="space-y-6">
            {data.map((segment) => (
                <div
                    key={segment.id}
                    className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                    <div className="mb-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                        <div>
                            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                                Speaker
                            </p>
                            <p className="font-semibold dark:text-white">{segment.speaker}</p>
                        </div>
                        <div>
                            <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                                Time
                            </p>
                            <p className="dark:text-white">
                                {formatTime(segment.start_time)}s - {formatTime(segment.end_time)}s
                                (Duration: {formatTime(segment.duration)}s)
                            </p>
                        </div>
                    </div>

                    <div className="mb-3">
                        <p className="mb-1 text-sm font-medium text-gray-500 dark:text-gray-400">
                            Original Audio
                        </p>
                        <audio
                            controls
                            src={`${originalAudioBaseUrl}${segment.file_path}?t=${Date.now()}`}
                            className="w-full"></audio>
                        <p className="mt-1 text-sm dark:text-gray-300">{segment.text}</p>
                    </div>

                    <hr className="my-4 border-gray-200 dark:border-gray-600" />

                    <div className="mb-3">
                        <p className="mb-1 text-sm font-medium text-gray-500 dark:text-gray-400">
                            Generated Audio
                        </p>
                        <audio
                            controls
                            src={`${generatedAudioBaseUrl}${segment.file_path}?t=${Date.now()}`}
                            className="w-full"></audio>
                        <textarea
                            value={editedTexts[segment.id] ?? ''}
                            onChange={(e) => handleTextChange(segment.id, e.target.value)}
                            className="mt-1 w-full rounded border border-gray-300 bg-white p-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                            rows={2}
                        />
                    </div>

                    <div className="mt-4 flex flex-col items-start gap-4 md:flex-row md:items-center">
                        <div className="w-full flex-1 md:max-w-md">
                            <Slider
                                label={`Speed: ${speeds[segment.id]?.toFixed(2) ?? 'N/A'}`}
                                value={speeds[segment.id] ?? 1}
                                onChange={(value) => handleSpeedChange(segment.id, value)}
                                minValue={0.5}
                                maxValue={3}
                                step={0.01}
                                className="w-full"
                            />
                        </div>
                        <button
                            onClick={() => handleRegenerate(segment.id)}
                            className="mt-2 rounded bg-orange-500 px-4 py-2 text-sm font-bold text-white hover:bg-orange-700 dark:bg-orange-600 dark:hover:bg-orange-800 md:mt-0">
                            Re-generate
                        </button>
                    </div>
                </div>
            ))}
        </div>
    )
}

export default AudioDisplay
