import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import DicomViewer, { type DicomViewerHandle } from "../components/DicomViewer";
import OverrideModal from "../components/OverrideModal";
import StoryNarrative from "../components/StoryNarrative";
import {
  fetchStudy,
  postGrounding,
  postOverride,
  sentencesFromAnalysis,
} from "../lib/api";
import { getFirstInstanceId } from "../lib/orthanc";
import { resolveHeatmapUrl } from "../lib/resolveUrls";

export default function StudyDetail() {
  const { id: recordId } = useParams<{ id: string }>();
  const viewerRef = useRef<DicomViewerHandle>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [study, setStudy] = useState<Awaited<ReturnType<typeof fetchStudy>> | null>(null);
  const [instanceId, setInstanceId] = useState<string | null>(null);
  const [orthancErr, setOrthancErr] = useState<string | null>(null);

  const [activeSentence, setActiveSentence] = useState<number | null>(null);
  const [groundingLoading, setGroundingLoading] = useState(false);
  const [groundingB64, setGroundingB64] = useState<string | null>(null);
  const [groundingErr, setGroundingErr] = useState<string | null>(null);

  const [overrideOpen, setOverrideOpen] = useState(false);

  const load = useCallback(async () => {
    if (!recordId) return;
    setLoading(true);
    setError(null);
    try {
      const s = await fetchStudy(recordId);
      setStudy(s);
      setOrthancErr(null);
      try {
        const iid = await getFirstInstanceId(s.orthanc_study_id);
        setInstanceId(iid);
        if (!iid) setOrthancErr("No series/instances found in Orthanc for this study id.");
      } catch (e) {
        setInstanceId(null);
        setOrthancErr(e instanceof Error ? e.message : String(e));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStudy(null);
    } finally {
      setLoading(false);
    }
  }, [recordId]);

  useEffect(() => {
    void load();
  }, [load]);

  const analysis = study?.analysis ?? {};
  const sentences = useMemo(() => sentencesFromAnalysis(analysis), [analysis]);

  const heatmapResolved = resolveHeatmapUrl(
    typeof analysis.heatmap_url === "string" ? analysis.heatmap_url : null
  );

  const pathologyEntries = useMemo(() => {
    const scores = analysis.pathology_scores;
    if (!scores || typeof scores !== "object") return [] as [string, number][];
    return Object.entries(scores as Record<string, number>).sort((a, b) => b[1] - a[1]);
  }, [analysis]);

  const structured = analysis.structured_findings;
  const structuredList = Array.isArray(structured) ? structured : [];

  const onSentenceSelect = async (index: number, sentence: string) => {
    setActiveSentence(index);
    setGroundingB64(null);
    setGroundingErr(null);
    setGroundingLoading(true);
    try {
      const blob = await viewerRef.current?.getViewportPngBlob();
      if (!blob) {
        throw new Error("Viewport not ready — wait for DICOM to load.");
      }
      const res = await postGrounding(sentence, blob);
      setGroundingB64(res.heatmap_png_base64);
    } catch (e) {
      setGroundingErr(e instanceof Error ? e.message : String(e));
    } finally {
      setGroundingLoading(false);
    }
  };

  const handleOverride = async (
    reason: string,
    corrected_labels: Record<string, unknown> | null
  ) => {
    if (!recordId) return;
    await postOverride(recordId, { reason, corrected_labels });
    await load();
  };

  if (!recordId) {
    return <p className="p-8 text-slate-400">Missing study id.</p>;
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <Link
          to="/"
          className="text-sm font-medium text-accent-glow hover:text-white"
        >
          ← Worklist
        </Link>
        {study && (
          <span className="rounded-full border border-ink-700 bg-ink-950 px-3 py-1 font-mono text-xs text-slate-400">
            {study.orthanc_study_id}
          </span>
        )}
        <button
          type="button"
          onClick={() => setOverrideOpen(true)}
          className="ml-auto rounded-lg border border-amber-700/50 bg-amber-950/30 px-3 py-1.5 text-sm text-amber-200 hover:bg-amber-950/50"
        >
          Override
        </button>
      </div>

      {loading && <p className="text-slate-400">Loading study…</p>}
      {error && (
        <div className="rounded-lg border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-200">
          {error}
        </div>
      )}

      {study && !loading && (
        <div className="grid gap-6 lg:grid-cols-2">
          <section className="space-y-3">
            <h2 className="text-lg font-semibold text-white">DICOM (Cornerstone + Orthanc)</h2>
            {orthancErr && (
              <p className="text-sm text-amber-300/90">{orthancErr}</p>
            )}
            <div className="relative rounded-xl border border-ink-800 bg-black/40 p-2">
              <DicomViewer ref={viewerRef} instanceId={instanceId} />
              {groundingB64 && (
                <img
                  src={`data:image/png;base64,${groundingB64}`}
                  alt="Phrase grounding heatmap"
                  className="ground-overlay pointer-events-none absolute inset-2 z-10 h-[calc(100%-1rem)] w-[calc(100%-1rem)] rounded-md object-fill"
                />
              )}
            </div>
            {groundingErr && (
              <p className="text-xs text-red-400">{groundingErr}</p>
            )}
            <p className="text-xs text-slate-500">
              Grounding sends a PNG export of the rendered viewport (not raw DICOM) to BioViL.
            </p>
          </section>

          <section className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-white">Case</h2>
              <dl className="mt-2 grid grid-cols-2 gap-2 text-sm">
                <dt className="text-slate-500">Patient</dt>
                <dd className="font-mono text-slate-200">{study.patient_id || "—"}</dd>
                <dt className="text-slate-500">Risk</dt>
                <dd className="font-mono text-teal-300">
                  {(study.risk_score * 100).toFixed(1)}%
                </dd>
                <dt className="text-slate-500">Status</dt>
                <dd>{study.status}</dd>
              </dl>
            </div>

            {pathologyEntries.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                  Pathology scores
                </h3>
                <ul className="mt-2 max-h-40 space-y-1 overflow-auto rounded-lg border border-ink-800 bg-ink-950/50 p-2 font-mono text-xs">
                  {pathologyEntries.map(([k, v]) => (
                    <li key={k} className="flex justify-between gap-2 text-slate-300">
                      <span>{k}</span>
                      <span className="text-accent-glow">{(v * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {structuredList.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                  Structured findings
                </h3>
                <pre className="mt-2 max-h-48 overflow-auto rounded-lg border border-ink-800 bg-ink-950/80 p-3 font-mono text-xs text-slate-300">
                  {JSON.stringify(structuredList, null, 2)}
                </pre>
              </div>
            )}

            {heatmapResolved && (
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                  Pipeline heatmap
                </h3>
                <a
                  href={heatmapResolved}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 block overflow-hidden rounded-lg border border-ink-800"
                >
                  <img src={heatmapResolved} alt="CXR heatmap" className="max-h-64 w-full object-contain bg-black" />
                </a>
                <p className="mt-1 text-xs text-slate-500">
                  Set <code className="font-mono">VITE_GATEWAY_PUBLIC_BASE</code> if this URL is on
                  the gateway host.
                </p>
              </div>
            )}

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                Narrative & grounding
              </h3>
              <div className="mt-2 rounded-xl border border-ink-800 bg-ink-950/40 p-4">
                <StoryNarrative
                  sentences={sentences}
                  activeIndex={activeSentence}
                  groundingLoading={groundingLoading}
                  onSelect={onSentenceSelect}
                />
              </div>
            </div>
          </section>
        </div>
      )}

      <OverrideModal
        open={overrideOpen}
        onClose={() => setOverrideOpen(false)}
        onSubmit={handleOverride}
      />
    </div>
  );
}
