'use client'

import useFetch from '@/utils/useFetch'
import React, { useEffect, useState } from 'react'
import { Status } from '../page'
import { baseUrl } from '@/utils/useFetch'
import AudioDisplay from '@/components/AudioDisplay'
import { useRouter } from 'next/navigation'

type Props = {}

const Page = (props: Props) => {
    const router = useRouter()
    const original_audio_base_url = `${baseUrl}/public/phase1/`
    const generated_audio_base_url = `${baseUrl}/public/phase4/`

    const { fetchData } = useFetch()
    const [modeList, setModeList] = useState<any>(null)
    const [selectedModels, setSelectedModels] = useState<{
        [key: string]: { sovits: string; gpt: string }
    }>({
        SPEAKER_00: { sovits: '', gpt: '' },
        SPEAKER_01: { sovits: '', gpt: '' },
    })
    const [finalResult, setFinalResult] = useState<any>(null)

    const [loading, setLoading] = useState<boolean>(false)
    const [uploadStatus, setUploadStatus] = useState<Status | null>(null)

    useEffect(() => {
        const getModelList = async () => {
            try {
                const result = await fetchData<any>('/phase4/model-list', {
                    method: 'GET',
                })
                setModeList(result)
            } catch (error) {
                console.error('Error fetching model list:', error)
            }
        }
        const getFinalResult = async () => {
            try {
                const result = await fetchData<any[]>('/phase4/result', {
                    method: 'GET',
                })
                if (Array.isArray(result)) {
                    result.sort((a, b) => a.start_time - b.start_time)
                }
                setFinalResult(result)
            } catch (error) {
                console.error('Error fetching final result:', error)
                setFinalResult(null)
            }
        }
        getModelList()
        getFinalResult()
    }, [])

    useEffect(() => {
        if (modeList) {
            const newSelectedModels = { ...selectedModels }
            Object.keys(modeList.name).forEach((speaker) => {
                const gptModels = modeList.model_list.gpt_model_list
                    .sort((a: string, b: string) => {
                        const getE = (str: string) => parseInt(str.match(/-?e(\d+)/)?.[1] || '0')
                        return getE(a) - getE(b)
                    })
                    .filter((model: string) => model.startsWith(`${speaker}_gpt`))
                const sovitsModels = modeList.model_list.sovits_model_list
                    .sort((a: string, b: string) => {
                        const getE = (str: string) => parseInt(str.match(/-?e(\d+)/)?.[1] || '0')
                        return getE(a) - getE(b)
                    })
                    .filter((model: string) => model.startsWith(`${speaker}_sovits`))

                if (sovitsModels.length > 0) {
                    newSelectedModels[speaker].sovits = sovitsModels[sovitsModels.length - 1]
                }
                if (gptModels.length > 0) {
                    newSelectedModels[speaker].gpt = gptModels[gptModels.length - 1]
                }
            })
            setSelectedModels(newSelectedModels)
        }
    }, [modeList])

    const handleGenerate = async () => {
        setLoading(true)
        setUploadStatus(null)
        const response = await fetchData<any>('/phase4/generate', {
            method: 'POST',
            body: JSON.stringify({
                selectedModels,
                ref_freeze: false,
            }),
        })

        if (response) {
            setUploadStatus({
                mode: 300,
                message: response.message || 'Generation started...',
            })

            const interval = setInterval(async () => {
                try {
                    const statusResponse = await fetchData<any>('/status/phase4', {
                        method: 'GET',
                    })

                    if (statusResponse) {
                        if (statusResponse.is_complete === true) {
                            clearInterval(interval)
                            setLoading(false)
                            setUploadStatus({ mode: 200, message: 'Generation complete!' })
                            window.location.reload()
                        } else {
                            setUploadStatus({
                                mode: 300,
                                message: statusResponse.message || 'Processing...',
                            })
                        }
                    } else {
                        clearInterval(interval)
                        setLoading(false)
                        setUploadStatus({ mode: 500, message: 'Failed to get status.' })
                    }
                } catch (error) {
                    console.error('Error fetching status:', error)
                    clearInterval(interval)
                    setLoading(false)
                    setUploadStatus({ mode: 500, message: 'Error fetching status.' })
                }
            }, 3000)
        } else {
            setUploadStatus({ mode: 500, message: 'Generation request failed' })
            setLoading(false)
        }
    }

    const handleToPhase5 = () => {
        router.push('/phase5')
    }

    if (!modeList) return <div className="p-4 dark:text-white">Loading model list...</div>

    return (
        <div className="container mx-auto p-4 dark:bg-gray-900 dark:text-white">
            <div className="mb-6 rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                <h1 className="mb-4 text-2xl font-bold">Model Selection & Generation</h1>
                <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                    {Object.keys(modeList.name).map((speaker) => (
                        <div key={speaker} className="mb-4 rounded border p-3 dark:border-gray-600">
                            <h2 className="mb-2 text-xl font-semibold">{speaker}</h2>
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                    SoVITS Model:
                                    <select
                                        value={selectedModels[speaker]?.sovits || ''}
                                        onChange={(e) => {
                                            const newSelectedModels = { ...selectedModels }
                                            newSelectedModels[speaker] = {
                                                ...newSelectedModels[speaker],
                                                sovits: e.target.value,
                                            }
                                            setSelectedModels(newSelectedModels)
                                        }}
                                        className="mt-1 block w-full rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white">
                                        <option value="">-- Select SoVITS --</option>
                                        {modeList.model_list.sovits_model_list
                                            .filter((model: string) =>
                                                model.startsWith(`${speaker}_sovits`)
                                            )
                                            .map((model: string) => (
                                                <option key={model} value={model}>
                                                    {model}
                                                </option>
                                            ))}
                                    </select>
                                </label>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                    GPT Model:
                                    <select
                                        value={selectedModels[speaker]?.gpt || ''}
                                        onChange={(e) => {
                                            const newSelectedModels = { ...selectedModels }
                                            newSelectedModels[speaker] = {
                                                ...newSelectedModels[speaker],
                                                gpt: e.target.value,
                                            }
                                            setSelectedModels(newSelectedModels)
                                        }}
                                        className="mt-1 block w-full rounded border border-gray-300 bg-white px-3 py-2 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white">
                                        <option value="">-- Select GPT --</option>
                                        {modeList.model_list.gpt_model_list
                                            .filter((model: string) =>
                                                model.startsWith(`${speaker}_gpt`)
                                            )
                                            .map((model: string) => (
                                                <option key={model} value={model}>
                                                    {model}
                                                </option>
                                            ))}
                                    </select>
                                </label>
                            </div>
                        </div>
                    ))}
                </div>
                <div className="mt-4 flex flex-wrap items-center justify-end gap-3">
                    {' '}
                    <button
                        onClick={handleGenerate}
                        disabled={loading}
                        className={`rounded px-4 py-2 font-bold text-white ${
                            loading
                                ? 'cursor-not-allowed bg-gray-500'
                                : 'bg-green-500 hover:bg-green-700 dark:bg-green-600 dark:hover:bg-green-800'
                        }`}>
                        {loading ? 'Generating...' : 'Generate Audio'}
                    </button>
                    <button
                        onClick={handleToPhase5}
                        disabled={loading}
                        className={`rounded px-4 py-2 font-bold text-white ${
                            loading
                                ? 'cursor-not-allowed bg-gray-500'
                                : 'bg-blue-500 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-800'
                        }`}>
                        {loading ? 'Generating...' : 'Continue to Phase 5'}
                    </button>
                </div>
                {uploadStatus && (
                    <div
                        className={`mt-4 rounded p-3 text-center text-sm ${
                            uploadStatus.mode >= 500
                                ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                : uploadStatus.mode === 200
                                  ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                  : 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                        }`}>
                        {uploadStatus.message}
                    </div>
                )}
            </div>

            <div className="mt-8">
                <h2 className="mb-4 text-xl font-bold">Generated Audio Results</h2>
                {loading && !finalResult && (
                    <p className="dark:text-gray-400">Waiting for generation to complete...</p>
                )}
                {!loading && finalResult ? (
                    <AudioDisplay
                        data={finalResult}
                        originalAudioBaseUrl={original_audio_base_url}
                        generatedAudioBaseUrl={generated_audio_base_url}
                        selectedModels={selectedModels}
                    />
                ) : !loading ? (
                    <p className="dark:text-gray-400">
                        No generated audio found. Select models and click 'Generate Audio'.
                    </p>
                ) : null}
            </div>

            {/* <p>{JSON.stringify(finalResult, null, 4)}</p> */}
        </div>
    )
}

export default Page
