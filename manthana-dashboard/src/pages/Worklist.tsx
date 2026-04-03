import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fetchWorklist, type WorklistItem } from "../lib/api";

function riskClass(score: number): string {
  if (score >= 0.75) return "bg-risk-crit/20 text-risk-crit border-risk-crit/40";
  if (score >= 0.5) return "bg-risk-high/20 text-risk-high border-risk-high/40";
  if (score >= 0.25) return "bg-risk-mid/20 text-risk-mid border-risk-mid/40";
  return "bg-risk-low/20 text-risk-low border-risk-low/40";
}

export default function Worklist() {
  const [items, setItems] = useState<WorklistItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchWorklist();
      setItems(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const sorted = useMemo(() => {
    return [...items].sort((a, b) => b.risk_score - a.risk_score);
  }, [items]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-ink-800 pb-6">
        <div>
          <p className="text-xs font-medium uppercase tracking-widest text-accent-glow">
            Manthana CXR
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-white">Worklist</h1>
          <p className="mt-2 max-w-xl text-sm text-slate-400">
            Studies ordered by model risk (highest first). Open a row for Cornerstone viewing,
            narrative sentences, and phrase grounding.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          className="rounded-lg border border-ink-700 bg-ink-900 px-4 py-2 text-sm font-medium text-white hover:border-accent/50"
        >
          Refresh
        </button>
      </header>

      {loading && <p className="text-slate-400">Loading worklist…</p>}
      {error && (
        <div className="rounded-lg border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-200">
          {error}
          <p className="mt-2 text-xs text-red-300/80">
            If the API requires auth, set a Bearer token in the header bar.
          </p>
        </div>
      )}

      {!loading && !error && sorted.length === 0 && (
        <p className="rounded-lg border border-dashed border-ink-700 bg-ink-950/50 p-8 text-center text-slate-400">
          No studies in the dashboard DB yet. Ingest via{" "}
          <code className="font-mono text-accent-glow">POST /internal/ingest</code>.
        </p>
      )}

      {!loading && sorted.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-ink-800 bg-ink-900/40 shadow-panel">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-ink-800 bg-ink-950/80 text-xs uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3 font-medium">Risk</th>
                <th className="px-4 py-3 font-medium">Patient</th>
                <th className="px-4 py-3 font-medium">Orthanc study</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Summary</th>
                <th className="px-4 py-3 font-medium">Updated</th>
                <th className="px-4 py-3 font-medium" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => (
                <tr
                  key={row.id}
                  className="border-b border-ink-800/80 transition-colors hover:bg-teal-950/20"
                >
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex min-w-[3.5rem] justify-center rounded-md border px-2 py-0.5 font-mono text-xs font-semibold ${riskClass(row.risk_score)}`}
                    >
                      {(row.risk_score * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-slate-300">
                    {row.patient_id || "—"}
                  </td>
                  <td className="max-w-[200px] truncate px-4 py-3 font-mono text-xs text-slate-400">
                    {row.orthanc_study_id}
                  </td>
                  <td className="px-4 py-3">
                    <span className="rounded bg-ink-800 px-2 py-0.5 text-xs text-slate-300">
                      {row.status}
                    </span>
                  </td>
                  <td className="max-w-xs truncate px-4 py-3 text-slate-400">
                    {row.screening_summary || "—"}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-xs text-slate-500">
                    {row.updated_at
                      ? new Date(row.updated_at).toLocaleString(undefined, {
                          dateStyle: "short",
                          timeStyle: "short",
                        })
                      : "—"}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Link
                      to={`/study/${row.id}`}
                      className="inline-flex rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-white hover:bg-accent-muted"
                    >
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
