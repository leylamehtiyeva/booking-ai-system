# Booking AI — website

Astro + React + Tailwind CSS. `/` is the landing page, `/demo` is the live demo
over the FastAPI backend (`POST /api/chat`, see `../app/api/main.py`).

```bash
npm install
npm run dev            # http://localhost:4321 — /api is proxied to http://127.0.0.1:8000
npm test               # Vitest: API client and demo flows on recorded responses
npm run check          # TypeScript
npm run build          # static site in dist/
```

Run the backend next to it: `uv run uvicorn app.api.main:app --port 8000` (repo root).

- The frontend only renders the public `TurnResult` contract (`src/lib/types.ts`);
  no matching, ranking or wording logic lives here.
- `src/mocks/turns.json` holds responses recorded from the real backend on the sample
  listings, offline (`npm run mocks`: scripted LLM boundaries, network blocked).
  Tests and the landing page use them; `PUBLIC_DEMO_MOCK=1 npm run dev` replays them
  in the demo for local work. A production build never includes the mock.
- Requires Node 20.3+. `astro check` needs Node 20.19+, so `npm run check` runs `tsc`.
