# Booking AI — website

Astro + React + Tailwind CSS. Node 22.22+ (`.nvmrc`).

- `/` — the portfolio landing page, fully static.
- `/demo` — the interactive demo over the FastAPI backend (`POST /api/chat`,
  see `../app/api/main.py`). It is **off in production builds** unless
  `PUBLIC_LIVE_DEMO=1` is set at build time; then `/demo` is a redirect to
  the recorded walkthrough and the chat client is not built at all.

```bash
npm ci
npm run dev            # http://localhost:4321 — live demo on; /api is proxied to http://127.0.0.1:8000
npm test               # Vitest: API client and demo flows on recorded responses
npm run check          # astro check + TypeScript
npm run build          # static portfolio site in dist/ (no backend needed)
npm run preview        # serve dist/ locally
```

Live demo locally: run the backend next to the dev server,
`uv run uvicorn app.api.main:app --port 8000` (repo root). Without a backend,
`PUBLIC_DEMO_MOCK=1 npm run dev` replays recorded responses (development only).
`PUBLIC_LIVE_DEMO=0 npm run dev` previews the static portfolio site.

To publish with the live demo later: deploy the backend, set
`PUBLIC_LIVE_DEMO=1` (and `PUBLIC_API_BASE_URL` if the API is on another
origin) for the build, and allow the site origin in `BOOKING_API_CORS_ORIGINS`.

- The frontend only renders the public `TurnResult` contract (`src/lib/types.ts`);
  no matching, ranking or wording logic lives here.
- `src/mocks/turns.json` holds responses recorded from the real backend on the sample
  listings, offline (`npm run mocks`: scripted LLM boundaries, network blocked).
  Tests and the landing page use them. `public/walkthrough/*.jpg` are screenshots
  of the demo replaying them (site header and replay telemetry left out of frame).
