/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: false,
    experimental: {
        serverActions: {
            bodySizeLimit: '50mb', // Increase body size limit to 50MB for video uploads
        },
    },
}

module.exports = nextConfig
