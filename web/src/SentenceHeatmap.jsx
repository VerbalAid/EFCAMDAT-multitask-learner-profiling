import { useMemo } from "react";

const ERROR_THRESHOLD = 0.03;

const HEAD_PALETTE = {
  cefr: { toward: [180, 110, 50], against: [50, 100, 160] },
  l1: { toward: [40, 140, 70], against: [100, 150, 140] },
  nat: { toward: [120, 70, 180], against: [100, 110, 130] },
};

/** Net toward-class attribution from API — do not re-derive from toward_mass. */
function signedScore(row) {
  if (row.signed_attribution != null) return row.signed_attribution;
  if (row.signed_mass != null) return row.signed_mass;
  return 0;
}

function isToward(row) {
  return signedScore(row) > 0;
}

function sentenceBackground(head, row, maxSigned) {
  const palette = HEAD_PALETTE[head] || HEAD_PALETTE.cefr;
  const toward = isToward(row);
  const rgb = toward ? palette.toward : palette.against;
  const mass = Math.abs(signedScore(row));
  const intensity = maxSigned > 0 ? Math.min(mass / maxSigned, 1) : 0;
  const scaled = toward ? 0.04 + intensity * 0.22 : 0.03 + intensity * 0.12;

  return {
    backgroundColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${scaled})`,
    borderColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${toward ? 0.12 + intensity * 0.22 : 0.1 + intensity * 0.15})`,
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

function SentenceBlock({ row, head, rank, maxSigned }) {
  const score = signedScore(row);
  const supporting = isToward(row);
  const lowSignal = Math.abs(score) < ERROR_THRESHOLD;
  const bg = sentenceBackground(head, row, maxSigned);
  const showPct = supporting && maxSigned > 0;
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
                title="Relative net toward-class attribution vs the strongest sentence in this head"
              >
                {Math.round((score / maxSigned) * 100)}%
              </span>
            )}
          </div>
        )}
        <HighlightedSentence sentence={row.sentence} tokens={row.tokens} head={head} />
      </div>
    </div>
  );
}

export default function SentenceHeatmap({ head, sentences, compact = false }) {
  const rows = useMemo(() => sentences || [], [sentences]);

  const maxSigned = useMemo(
    () => (rows.length ? Math.max(0, ...rows.map((s) => signedScore(s))) : 0),
    [rows]
  );

  if (!rows.length) {
    return null;
  }

  const hasSignal = maxSigned > 0;
  let rank = 0;

  return (
    <div className={`sentence-list sentence-list--${head}${compact ? " sentence-list--compact" : ""}`}>
      {hasSignal && !compact && (
        <p className="sentence-list__legend">
          Top supporting sentences · % = relative net attribution · highlights = key tokens
        </p>
      )}
      {rows.map((row) => {
        if (isToward(row) && Math.abs(signedScore(row)) >= ERROR_THRESHOLD) {
          rank += 1;
        }
        return (
          <SentenceBlock
            key={row.sentence}
            row={row}
            head={head}
            rank={rank}
            maxSigned={maxSigned}
          />
        );
      })}
    </div>
  );
}
