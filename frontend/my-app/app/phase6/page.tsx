'use client'

import React, { useEffect, useState } from 'react'
import { baseUrl } from '@/utils/useFetch'

const Page = () => {
    const [showVideo, setShowVideo] = useState(false)

    useEffect(() => {
        setShowVideo(true)
    }, [])

    return (
        <div className="container mx-auto px-4 py-6 text-gray-900 dark:text-gray-100">
            <h1 className="text-2xl font-bold mb-4">Phase 6: Final Video Generated</h1>
            {showVideo && (
                <video
                    className="w-full h-auto rounded-lg shadow-lg"
                    controls
                    src={`${baseUrl}/public/phase5/final_video.mp4`}>
                    Your browser does not support the video tag.
                </video>
            )}
        </div>
    )
}

export default Page
