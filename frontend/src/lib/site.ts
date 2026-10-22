export const REPO_URL = "https://github.com/leylamehtiyeva/booking-ai-system";

/**
 * Live demo: the /demo chat over the FastAPI backend (paid LLM calls).
 * Production builds are a static portfolio site unless PUBLIC_LIVE_DEMO=1 is
 * set at build time; `astro dev` keeps the demo on (PUBLIC_LIVE_DEMO=0 turns
 * it off to preview the static site). When off, /demo redirects to the home
 * page. The demo code is unchanged either way.
 */
const flag = import.meta.env.PUBLIC_LIVE_DEMO;
export const LIVE_DEMO = flag === "1" || (import.meta.env.DEV && flag !== "0");
