import { useState } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  onSubmit: (reason: string, correctedLabels: Record<string, unknown> | null) => Promise<void>;
};

export default function OverrideModal({ open, onClose, onSubmit }: Props) {
  const [reason, setReason] = useState("");
  const [labelsJson, setLabelsJson] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!open) return null;

  const handleSubmit = async () => {
    setErr(null);
    let corrected: Record<string, unknown> | null = null;
    if (labelsJson.trim()) {
      try {
        corrected = JSON.parse(labelsJson) as Record<string, unknown>;
      } catch {
        setErr("Corrected labels must be valid JSON or empty.");
        return;
      }
    }
    setBusy(true);
    try {
      await onSubmit(reason, corrected);
      setReason("");
      setLabelsJson("");
      onClose();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl border border-ink-700 bg-ink-900 p-6 shadow-panel">
        <h2 className="text-lg font-semibold text-white">Radiologist override</h2>
        <p className="mt-1 text-sm text-slate-400">
          Submits to the dashboard API and optional webhook (
          <code className="font-mono text-xs text-accent-glow">OVERRIDE_WEBHOOK_URL</code>
          ).
        </p>
        <label className="mt-4 block text-xs font-medium uppercase tracking-wide text-slate-500">
          Reason
        </label>
        <textarea
          className="mt-1 w-full rounded-lg border border-ink-700 bg-ink-950 px-3 py-2 text-sm text-white placeholder:text-ink-700 focus:border-accent focus:outline-none"
          rows={3}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Clinical or QA reason…"
        />
        <label className="mt-3 block text-xs font-medium uppercase tracking-wide text-slate-500">
          Corrected labels (optional JSON)
        </label>
        <textarea
          className="mt-1 w-full rounded-lg border border-ink-700 bg-ink-950 px-3 py-2 font-mono text-xs text-white placeholder:text-ink-700 focus:border-accent focus:outline-none"
          rows={5}
          value={labelsJson}
          onChange={(e) => setLabelsJson(e.target.value)}
          placeholder='{"pathology_scores": {"Atelectasis": 0.1}}'
        />
        {err && <p className="mt-2 text-sm text-red-400">{err}</p>}
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            className="rounded-lg px-4 py-2 text-sm text-slate-400 hover:text-white"
            onClick={onClose}
            disabled={busy}
          >
            Cancel
          </button>
          <button
            type="button"
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-muted disabled:opacity-50"
            onClick={() => void handleSubmit()}
            disabled={busy}
          >
            {busy ? "Saving…" : "Confirm override"}
          </button>
        </div>
      </div>
    </div>
  );
}
