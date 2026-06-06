import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/predict": "http://127.0.0.1:8000",
      "/errors": "http://127.0.0.1:8000",
      "/upload": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
