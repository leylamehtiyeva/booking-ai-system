import { useEffect, useState } from "react";
import type { ChatTransport } from "../../lib/api";
import DemoApp from "./DemoApp";

// The real FastAPI backend, always - except in local development with
// PUBLIC_DEMO_MOCK=1, where recorded responses are replayed. import.meta.env.DEV
// is false in a production build, so the mock module is not even bundled.
const MOCK = import.meta.env.DEV && import.meta.env.PUBLIC_DEMO_MOCK === "1";

export default function DemoRoot() {
  const [mock, setMock] = useState<ChatTransport | null>(null);

  useEffect(() => {
    if (MOCK) import("../../lib/mockTransport").then((m) => setMock(m.mockTransport()));
  }, []);

  if (MOCK && !mock) return null;
  return <DemoApp transport={mock ?? undefined} mock={MOCK} />;
}
