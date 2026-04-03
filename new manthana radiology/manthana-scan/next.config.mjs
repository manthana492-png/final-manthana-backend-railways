/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
      { protocol: "http", hostname: "127.0.0.1" },
    ],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Cornerstone3D WASM codecs reference Node.js 'fs' — stub it for browser
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };

      // ESM-only packages: tell webpack to use "import" condition in exports map
      config.resolve.conditionNames = ["import", "module", "browser", "default"];
    }

    // Ensure ESM-only packages resolve correctly
    config.resolve.extensionAlias = {
      ".js": [".ts", ".tsx", ".js", ".jsx"],
    };

    return config;
  },
  // Transpile Cornerstone3D packages (ESM-only)
  transpilePackages: [
    "@cornerstonejs/core",
    "@cornerstonejs/tools",
    "@cornerstonejs/dicom-image-loader",
    "@cornerstonejs/codec-charls",
    "@cornerstonejs/codec-libjpeg-turbo-8bit",
    "@cornerstonejs/codec-openjpeg",
    "@cornerstonejs/codec-openjph",
  ],
};

export default nextConfig;
