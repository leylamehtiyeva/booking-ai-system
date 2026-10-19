// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwindcss from "@tailwindcss/vite";
import { loadEnv } from "vite";

// The demo calls the FastAPI backend at /api. In development the Vite proxy
// forwards it to uvicorn, so the browser stays same-origin (no CORS needed).
const API_TARGET = process.env.BOOKING_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

/**
 * Live demo switch - the same rule as LIVE_DEMO in src/lib/site.ts: on in
 * `astro dev`, off in builds unless PUBLIC_LIVE_DEMO=1. When off, /demo is a
 * redirect to the recorded walkthrough and the chat client is not built.
 * @returns {import("astro").AstroIntegration}
 */
function liveDemoRoute() {
  return {
    name: "booking-ai:live-demo-route",
    hooks: {
      "astro:config:setup": ({ command, injectRoute, updateConfig }) => {
        const mode = command === "dev" ? "development" : "production";
        const flag = loadEnv(mode, process.cwd(), "PUBLIC_").PUBLIC_LIVE_DEMO;
        const live = flag === "1" || (command === "dev" && flag !== "0");
        if (live) {
          injectRoute({ pattern: "/demo", entrypoint: "./src/routes/demo.astro" });
        } else {
          updateConfig({ redirects: { "/demo": "/#walkthrough" } });
        }
      },
    },
  };
}

export default defineConfig({
  integrations: [react(), liveDemoRoute()],
  devToolbar: { enabled: false },
  vite: {
    plugins: [tailwindcss()],
    server: { proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
  },
});
