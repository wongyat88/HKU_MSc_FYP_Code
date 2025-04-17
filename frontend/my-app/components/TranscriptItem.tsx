'use client'

import React from 'react'

interface TranscriptItemProps {
    item: {
        id: number
        speaker: string
        start_time: number
        end_time: number
        text: string // Added text property
        translated_text: string
    }
    onTextChange: (id: number, newText: string) => void
}

// Define a list of colors for speakers
const speakerColors = [
    'text-red-600 dark:text-red-400',
    'text-green-600 dark:text-green-400',
    'text-blue-600 dark:text-blue-400',
    'text-yellow-600 dark:text-yellow-400',
    'text-purple-600 dark:text-purple-400',
    'text-pink-600 dark:text-pink-400',
    'text-indigo-600 dark:text-indigo-400',
    'text-teal-600 dark:text-teal-400',
]

const getSpeakerColor = (speaker: string): string => {
    const match = speaker.match(/_(\d+)$/) // Extract number from SPEAKER_XX
    if (match && match[1]) {
        const speakerIndex = parseInt(match[1], 10)
        return speakerColors[speakerIndex % speakerColors.length]
    }
    return 'text-gray-600 dark:text-gray-400' // Default color if pattern doesn't match
}

const TranscriptItem: React.FC<TranscriptItemProps> = ({ item, onTextChange }) => {
    const formatTime = (time: number) => time.toFixed(2)
    const speakerColorClass = getSpeakerColor(item.speaker)

    return (
        <div className="mb-4 rounded border border-gray-300 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
            <div className="mb-2 flex items-center justify-between">
                <span className={`font-semibold ${speakerColorClass}`}>{item.speaker}</span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                    {formatTime(item.start_time)}s - {formatTime(item.end_time)}s
                </span>
            </div>
            {/* Display original text */}
            <p className="mb-2 text-sm text-gray-600 dark:text-gray-400">
                <span className="font-medium">Original:</span> {item.text}
            </p>
            {/* Editable translated text */}
            <textarea
                value={item.translated_text}
                onChange={(e) => onTextChange(item.id, e.target.value)}
                className="w-full rounded border border-gray-300 bg-gray-50 p-2 text-gray-900 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500"
                rows={2}
                aria-label={`Translated text for ${item.speaker}`}
            />
        </div>
    )
}

export default TranscriptItem
