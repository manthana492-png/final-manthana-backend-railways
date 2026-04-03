import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const orthancAuth = process.env.VITE_ORTHANC_PROXY_AUTH || "manthana:manthana-pacs-secret";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3040,
    proxy: {
      "/api/dashboard": {
        target: "http://127.0.0.1:8037",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/dashboard/, ""),
      },
      "/api/biovil": {
        target: "http://127.0.0.1:8038",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/biovil/, ""),
      },
      "/orthanc": {
        target: "http://127.0.0.1:8042",
        changeOrigin: true,
        auth: orthancAuth,
        rewrite: (p) => p.replace(/^\/orthanc/, ""),
      },
    },
  },
});
