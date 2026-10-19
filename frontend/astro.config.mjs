// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";

// The demo calls the FastAPI backend at /api. In development the Vite proxy
// forwards it to uvicorn, so the browser stays same-origin (no CORS needed).
const API_TARGET = process.env.BOOKING_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  integrations: [react()],
  devToolbar: { enabled: false },
  vite: {
    plugins: [tailwindcss()],
    server: { proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
  },
});
