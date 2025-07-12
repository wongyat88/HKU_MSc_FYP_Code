import React, { useState } from 'react'
import { Select, SelectItem } from '@heroui/select'
import { Textarea } from '@heroui/input'
import { baseUrl } from '@/utils/useFetch'

interface AudioCardProps {
    item: {
        id: number
        file_path: string
        start_time: number
        end_time: number
        speaker: string
        text: string
    }
    index: number
    speakers: string[]
    onTextChange: (index: number, newText: string) => void
    onSpeakerChange?: (index: number, newSpeaker: string) => void
    onSelectionChange?: (id: number, isSelected: boolean) => void
    isSelected?: boolean
}

const AudioCard: React.FC<AudioCardProps> = ({
    item,
    index,
    speakers,
    onTextChange,
    onSpeakerChange,
    onSelectionChange,
    isSelected = false,
}) => {
    const [selected, setSelected] = useState(isSelected)

    const handleToggleSelection = () => {
        const newSelectedState = !selected
        setSelected(newSelectedState)

        // Notify parent component about selection change
        if (onSelectionChange) {
            onSelectionChange(item.id, newSelectedState)
        }
    }

    return (
        <div
            className={`rounded-lg shadow-md p-4 mb-4 border transition-colors duration-200 cursor-pointer
                ${
                    selected
                        ? 'bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700'
                        : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'
                }`}
            onClick={handleToggleSelection}
            aria-label={`Audio segment2 ${index + 1}`}>
            <div className="space-y-3">
                {/* Selection Indicator */}
                <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        ID: {item.id}
                    </span>
                    <span
                        className={`text-sm ${selected ? 'text-blue-600 dark:text-blue-400 font-medium' : 'text-gray-500 dark:text-gray-400'}`}>
                        {selected ? 'Selected' : 'Click to select'}
                    </span>
                </div>

                {/* Audio Player */}
                <div className="w-full" onClick={(e) => e.stopPropagation()}>
                    <audio controls className="w-full focus:outline-none">
                        <source
                            src={`${baseUrl}/public/phase1/${item.file_path}`}
                            type="audio/wav"
                        />
                        Your browser does not support the audio element.
                    </audio>
                </div>

                {/* Timing Information */}
                <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                    <span>
                        Start: <span className="font-medium">{item.start_time.toFixed(2)}s</span>
                    </span>
                    <span>
                        End: <span className="font-medium">{item.end_time.toFixed(2)}s</span>
                    </span>
                </div>

                {/* Speaker Selection */}
                <div onClick={(e) => e.stopPropagation()}>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Speaker
                    </label>
                    <Select
                        className="w-full"
                        defaultSelectedKeys={[item.speaker]}
                        aria-label="Select speaker"
                        onChange={(key: any) =>
                            onSpeakerChange && onSpeakerChange(index, key.target.value)
                        }>
                        {speakers.map((speaker) => (
                            <SelectItem key={speaker}>{speaker}</SelectItem>
                        ))}
                    </Select>
                </div>

                {/* Transcript Text */}
                <div onClick={(e) => e.stopPropagation()}>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Transcript
                    </label>
                    <Textarea
                        className="w-full min-h-[80px] resize-y"
                        defaultValue={item.text}
                        onChange={(e) => onTextChange(index, e.target.value)}
                        placeholder="Transcript text..."
                    />
                </div>
            </div>
        </div>
    )
}

export default AudioCard
