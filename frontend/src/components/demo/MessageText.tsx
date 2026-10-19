import { Fragment, type ReactNode } from "react";

// Minimal, safe rendering of the assistant's plain text: line breaks,
// "- " bullets, **bold** and [title](https://...) links. No HTML is injected.
const INLINE = /(\*\*[^*]+\*\*|\[[^\]]+\]\((https?:\/\/[^)\s]+)\))/g;

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const index = match.index ?? 0;
    if (index > last) out.push(text.slice(last, index));
    const token = match[0];
    if (token.startsWith("**")) {
      out.push(<strong key={index} className="font-semibold text-ink">{token.slice(2, -2)}</strong>);
    } else {
      const title = token.slice(1, token.indexOf("]"));
      out.push(
        <a key={index} href={match[2]} target="_blank" rel="noopener noreferrer" className="underline decoration-line-strong underline-offset-4 hover:decoration-accent">
          {title}
        </a>,
      );
    }
    last = index + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function MessageText({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/);
  return (
    <div className="space-y-3">
      {blocks.map((block, b) => {
        const lines = block.split("\n");
        const bullets = lines.every((line) => /^\s*[-•]\s+/.test(line));
        if (bullets) {
          return (
            <ul key={b} className="list-disc space-y-1 pl-5 marker:text-line-strong">
              {lines.map((line, i) => <li key={i}>{inline(line.replace(/^\s*[-•]\s+/, ""))}</li>)}
            </ul>
          );
        }
        return (
          <p key={b}>
            {lines.map((line, i) => (
              <Fragment key={i}>
                {i > 0 && <br />}
                {inline(line)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
