type Props = {
  sentences: string[];
  activeIndex: number | null;
  groundingLoading: boolean;
  onSelect: (index: number, sentence: string) => void;
};

export default function StoryNarrative({
  sentences,
  activeIndex,
  groundingLoading,
  onSelect,
}: Props) {
  if (!sentences.length) {
    return (
      <p className="text-sm text-slate-400">
        No narrative available yet. Ingest analysis with <code className="font-mono">narrative</code>{" "}
        or impression fields.
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
        Click a sentence — BioViL phrase grounding
      </p>
      <ul className="space-y-2">
        {sentences.map((s, i) => {
          const active = activeIndex === i;
          return (
            <li key={`${i}-${s.slice(0, 24)}`}>
              <button
                type="button"
                disabled={groundingLoading}
                onClick={() => onSelect(i, s)}
                className={`w-full rounded-lg border px-3 py-2 text-left text-sm leading-relaxed transition-colors ${
                  active
                    ? "border-accent bg-teal-950/40 text-white"
                    : "border-ink-700 bg-ink-950/50 text-slate-200 hover:border-accent/50"
                } ${groundingLoading && !active ? "opacity-60" : ""}`}
              >
                {s}
              </button>
            </li>
          );
        })}
      </ul>
      {groundingLoading && (
        <p className="text-xs text-accent-glow">Computing saliency map…</p>
      )}
    </div>
  );
}
