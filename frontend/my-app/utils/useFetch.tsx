'use client'

import { useState } from 'react'
import Swal from 'sweetalert2'

interface FetchOptions {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
    headers?: Record<string, string>
    body?: FormData | string | Record<string, any>
}

export const baseUrl = 'http://127.0.0.1:8000'

export const useFetch = () => {
    const [loading, setLoading] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    /**
     * Make a fetch request with error handling using SweetAlert2
     * @param url The URL to fetch
     * @param options Fetch options
     * @returns The response data if successful
     */
    const fetchData = async <T,>(url: string, options: FetchOptions = {}): Promise<T | null> => {
        setLoading(true)
        setError(null)

        try {
            // Prepare the request options
            const fetchOptions: any = {
                method: options.method || 'GET',
                headers: options.headers || {},
            }

            // Handle the request body based on its type
            if (options.body) {
                if (options.body instanceof FormData) {
                    fetchOptions.body = options.body
                } else {
                    if (!fetchOptions.headers['Content-Type']) {
                        fetchOptions.headers['Content-Type'] = 'application/json'
                    }
                    fetchOptions.body = options.body
                }
            }

            const response = await fetch(baseUrl + url, fetchOptions)

            // Handle non-200 responses
            if (!response.ok) {
                const errorText = await response.text()
                let errorMessage = `Error: ${response.status} ${response.statusText}`

                try {
                    // Try to parse error response as JSON
                    const errorJson = JSON.parse(errorText)
                    if (errorJson.message || errorJson.error) {
                        errorMessage = errorJson.message || errorJson.error
                    }
                } catch (e) {
                    // If not JSON, use the text response if available
                    if (errorText) {
                        errorMessage += ` - ${errorText}`
                    }
                }

                // Display error with SweetAlert2
                Swal.fire({
                    title: 'Error',
                    text: errorMessage,
                    icon: 'error',
                    confirmButtonText: 'OK',
                })

                setError(errorMessage)
                return null
            }

            // Parse response
            const data = await response.json()
            return data as T
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'

            // Display error with SweetAlert2
            Swal.fire({
                title: 'Error',
                text: errorMessage,
                icon: 'error',
                confirmButtonText: 'OK',
            })

            setError(errorMessage)
            return null
        } finally {
            setLoading(false)
        }
    }

    return { fetchData, loading, error }
}

export default useFetch
