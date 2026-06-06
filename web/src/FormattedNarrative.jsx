import { useMemo } from "react";

function stripListPrefix(line) {
  return line.replace(/^[-*•]\s+/, "").replace(/^\d+[.)]\s+/, "");
}

function normalizeText(text) {
  return text.replace(/\s+/g, " ").trim();
}

function isBulletLine(line) {
  return /^[-*•]\s/.test(line);
}

function isNumberedLine(line) {
  return /^\d+[.)]\s/.test(line);
}

function splitNumberedRunOn(text) {
  const parts = text
    .split(/(?=\d+[.)]\s)/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length <= 1) return null;
  return parts.map(stripListPrefix).map(normalizeText);
}

function parseNarrative(text) {
  const cleaned = text.trim();
  if (!cleaned) return [];

  const blocks = cleaned
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);

  const sections = [];

  for (const block of blocks) {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length > 1 && lines.every((line) => isBulletLine(line) || isNumberedLine(line))) {
      sections.push({
        type: "list",
        items: lines.map(stripListPrefix).map(normalizeText),
      });
      continue;
    }

    const firstBullet = lines.findIndex((line) => isBulletLine(line));
    if (firstBullet > 0) {
      sections.push({
        type: "mixed",
        intro: normalizeText(lines.slice(0, firstBullet).join(" ")),
        items: lines.slice(firstBullet).map(stripListPrefix).map(normalizeText),
      });
      continue;
    }

    const numberedStart = block.search(/\d+[.)]\s/);
    if (numberedStart > 0) {
      const intro = normalizeText(block.slice(0, numberedStart).replace(/:\s*$/, ""));
      const items = splitNumberedRunOn(block.slice(numberedStart)) || [];
      sections.push({ type: "mixed", intro, items });
      continue;
    }

    const inlineNumbered = splitNumberedRunOn(block);
    if (inlineNumbered && inlineNumbered.length > 1) {
      sections.push({ type: "list", items: inlineNumbered });
      continue;
    }

    if (lines.length === 1 && (isBulletLine(lines[0]) || isNumberedLine(lines[0]))) {
      sections.push({
        type: "list",
        items: [normalizeText(stripListPrefix(lines[0]))],
      });
      continue;
    }

    sections.push({ type: "paragraph", text: normalizeText(lines.join(" ")) });
  }

  return sections;
}

export default function FormattedNarrative({ text, className = "" }) {
  const sections = useMemo(() => parseNarrative(text), [text]);
  if (!sections.length) return null;

  return (
    <div className={`formatted-narrative ${className}`.trim()}>
      {sections.map((block, i) => {
        if (block.type === "list") {
          return (
            <ul key={i} className="formatted-narrative__list">
              {block.items.map((item, j) => (
                <li key={j}>{item}</li>
              ))}
            </ul>
          );
        }
        if (block.type === "mixed") {
          return (
            <div key={i} className="formatted-narrative__section">
              {block.intro && <p className="formatted-narrative__para">{block.intro}</p>}
              <ul className="formatted-narrative__list">
                {block.items.map((item, j) => (
                  <li key={j}>{item}</li>
                ))}
              </ul>
            </div>
          );
        }
        return (
          <p key={i} className="formatted-narrative__para">
            {block.text}
          </p>
        );
      })}
    </div>
  );
}
