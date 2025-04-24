'use client'

import useFetch from '@/utils/useFetch'
import React, { useEffect, useState } from 'react'
import { Button } from '@heroui/button'

type Props = {}

const Page = (props: Props) => {
    const { fetchData } = useFetch()
    const [modeList, setModeList] = useState<any>(null) // Changed to any since modeList is an object, not an array
    const [selectedModels, setSelectedModels] = useState<{
        [key: string]: { sovits: string; gpt: string }
    }>({
        SPEAKER_00: { sovits: '', gpt: '' },
        SPEAKER_01: { sovits: '', gpt: '' },
    })

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
        getModelList()
    }, [])

    useEffect(() => {
        // Auto-select the last option for each speaker
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

                newSelectedModels[speaker].sovits = sovitsModels[sovitsModels.length - 1]
                newSelectedModels[speaker].gpt = gptModels[gptModels.length - 1]
            })
            setSelectedModels(newSelectedModels)
        }
    }, [modeList])

    const handleSave = async () => {
        try {
            const response = await fetchData<any>('/phase4/generate', {
                method: 'POST',
                body: JSON.stringify({
                    selectedModels,
                    ref_freeze: false,
                }),
            })
            console.log('Test gen:', response)
        } catch (error) {
            console.error('Error testing generation:', error)
        }
    }

    if (!modeList) return <div>Loading...</div>

    return (
        <div className="container mx-auto p-4 dark:bg-gray-900 dark:text-white">
            <div className="mb-4 flex items-center justify-between">
                <h1 className="text-2xl font-bold">Audio Generation</h1>
                <button
                    onClick={handleSave}
                    className="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-800">
                    Save Changes
                </button>
            </div>

            {Object.keys(modeList.name).map((speaker, index) => (
                <div key={index} className="mb-4">
                    <h2 className="text-xl font-bold">{speaker}</h2>
                    <div className="flex">
                        <select
                            value={selectedModels[speaker].sovits}
                            onChange={(e) => {
                                const newSelectedModels = { ...selectedModels }
                                newSelectedModels[speaker].sovits = e.target.value
                                setSelectedModels(newSelectedModels)
                            }}
                            className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-2 mr-4">
                            {modeList.model_list.sovits_model_list
                                .filter((model: string) => model.startsWith(`${speaker}_sovits`))
                                .map((model: string) => (
                                    <option key={model} value={model}>
                                        {model}
                                    </option>
                                ))}
                        </select>
                        <select
                            value={selectedModels[speaker].gpt}
                            onChange={(e) => {
                                const newSelectedModels = { ...selectedModels }
                                newSelectedModels[speaker].gpt = e.target.value
                                setSelectedModels(newSelectedModels)
                            }}
                            className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-3 py-2">
                            {modeList.model_list.gpt_model_list
                                .filter((model: string) => model.startsWith(`${speaker}_gpt`))
                                .map((model: string) => (
                                    <option key={model} value={model}>
                                        {model}
                                    </option>
                                ))}
                        </select>
                    </div>
                </div>
            ))}

            {
                JSON.stringify(selectedModels, null, 2) // Display selected models for debugging
            }
        </div>
    )
}

export default Page
