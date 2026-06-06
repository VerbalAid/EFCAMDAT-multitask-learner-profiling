import { useMemo } from "react";

const ERROR_THRESHOLD = 0.03;

const HEAD_PALETTE = {
  cefr: { toward: [180, 110, 50], against: [50, 100, 160] },
  l1: { toward: [40, 140, 70], against: [100, 150, 140] },
  nat: { toward: [120, 70, 180], against: [100, 110, 130] },
};

function sentenceBackground(head, row, maxToward) {
  const palette = HEAD_PALETTE[head] || HEAD_PALETTE.cefr;
  const toward = row.direction !== "negative";
  const rgb = toward ? palette.toward : palette.against;
  const mass = row.toward_mass ?? row.attribution ?? 0;
  const intensity = maxToward > 0 ? Math.min(mass / maxToward, 1) : 0;
  const scaled = 0.04 + intensity * 0.22;

  if (toward) {
    return {
      backgroundColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${scaled})`,
      borderColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.12 + intensity * 0.22})`,
    };
  }
  return {
    backgroundColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.03 + intensity * 0.12})`,
    borderColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.1 + intensity * 0.15})`,
  };
}

function tokenStyle(head, tokenAttribution, maxTokenAttr) {
  const palette = HEAD_PALETTE[head] || HEAD_PALETTE.cefr;
  const rgb = palette.toward;
  const rel = maxTokenAttr > 0 ? Math.min(tokenAttribution / maxTokenAttr, 1) : 0;
  const alpha = 0.14 + rel * 0.22;
  return {
    backgroundColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`,
    borderRadius: "3px",
    padding: "0 2px",
    boxDecorationBreak: "clone",
    WebkitBoxDecorationBreak: "clone",
  };
}

function buildSegments(sentence, tokens) {
  if (!tokens?.length) {
    return [{ text: sentence, highlight: null }];
  }
  const marked = tokens.flatMap((t) =>
    Array.from({ length: t.end - t.start }, (_, i) => ({
      index: t.start + i,
      token: t,
    }))
  );
  const byIndex = new Map(marked.map((m) => [m.index, m.token]));

  const segments = [];
  let i = 0;
  while (i < sentence.length) {
    const tok = byIndex.get(i);
    if (tok) {
      segments.push({ text: sentence.slice(tok.start, tok.end), highlight: tok });
      i = tok.end;
    } else {
      let j = i + 1;
      while (j < sentence.length && !byIndex.has(j)) j += 1;
      segments.push({ text: sentence.slice(i, j), highlight: null });
      i = j;
    }
  }
  return segments;
}

function HighlightedSentence({ sentence, tokens, head }) {
  const segments = useMemo(() => buildSegments(sentence, tokens), [sentence, tokens]);
  const maxTokenAttr = useMemo(
    () => (tokens?.length ? Math.max(...tokens.map((t) => t.attribution || 0)) : 0),
    [tokens]
  );
  return (
    <p className="sentence-block__text">
      {segments.map((seg, i) =>
        seg.highlight ? (
          <mark
            key={i}
            className={`token-mark token-mark--${head}`}
            style={tokenStyle(head, seg.highlight.attribution, maxTokenAttr)}
            title="Supports this prediction"
          >
            {seg.text}
          </mark>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
    </p>
  );
}

function SentenceBlock({ row, head, rank, maxToward }) {
  const lowSignal = (row.toward_mass ?? row.attribution ?? 0) < ERROR_THRESHOLD;
  const supporting = row.direction !== "negative";
  const bg = sentenceBackground(head, row, maxToward);
  const showPct = supporting && maxToward > 0;
  const showRank = supporting && rank <= 3 && !lowSignal;
  const showMeta = showPct || showRank || !supporting;

  return (
    <div className={`sentence-block-wrap sentence-block-wrap--${head}`}>
      <div
        className={`sentence-block${lowSignal ? " sentence-block--faint" : ""}${supporting ? " sentence-block--toward" : " sentence-block--against"}`}
        style={bg}
      >
        {showMeta && (
          <div className="sentence-block__meta">
            <div className="sentence-block__meta-left">
              {showRank && <span className="sentence-block__rank">{rank}</span>}
              {!supporting && (
                <span className="sentence-block__badge sentence-block__badge--against">against</span>
              )}
            </div>
            {showPct && (
              <span
                className="sentence-block__pct"
                title="Relative SHAP strength vs the strongest sentence in this head"
              >
                {Math.round(((row.toward_mass ?? row.attribution ?? 0) / maxToward) * 100)}%
              </span>
            )}
          </div>
        )}
        <HighlightedSentence sentence={row.sentence} tokens={row.tokens} head={head} />
      </div>
    </div>
  );
}

export default function SentenceHeatmap({ head, sentences }) {
  const rows = useMemo(() => sentences || [], [sentences]);

  const maxToward = useMemo(
    () => (rows.length ? Math.max(...rows.map((s) => s.toward_mass ?? s.attribution ?? 0)) : 0),
    [rows]
  );

  if (!rows.length) {
    return null;
  }

  const hasSignal = maxToward > 0;
  let rank = 0;

  return (
    <div className={`sentence-list sentence-list--${head}`}>
      {hasSignal && (
        <p className="sentence-list__legend">
          Top supporting sentences · % = relative SHAP strength · highlights = key tokens
        </p>
      )}
      {rows.map((row) => {
        if (row.direction !== "negative" && (row.toward_mass ?? row.attribution ?? 0) >= ERROR_THRESHOLD) {
          rank += 1;
        }
        return (
          <SentenceBlock
            key={row.sentence}
            row={row}
            head={head}
            rank={rank}
            maxToward={maxToward}
          />
        );
      })}
    </div>
  );
}
