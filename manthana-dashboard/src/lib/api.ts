import { authHeaders, biovilApiBase, dashboardApiBase } from "./config";

export type WorklistItem = {
  id: string;
  orthanc_study_id: string;
  patient_id: string | null;
  risk_score: number;
  status: string;
  screening_summary: string | null;
  updated_at: string | null;
};

export type StudyDetail = {
  id: string;
  orthanc_study_id: string;
  patient_id: string | null;
  risk_score: number;
  status: string;
  analysis: Record<string, unknown>;
};

export async function fetchWorklist(limit = 200): Promise<WorklistItem[]> {
  const r = await fetch(`${dashboardApiBase}/worklist?limit=${limit}`, {
    headers: { ...authHeaders() },
  });
  if (!r.ok) throw new Error(await r.text());
  const data = (await r.json()) as { items: WorklistItem[] };
  return data.items;
}

export async function fetchStudy(recordId: string): Promise<StudyDetail> {
  const r = await fetch(`${dashboardApiBase}/studies/${encodeURIComponent(recordId)}`, {
    headers: { ...authHeaders() },
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<StudyDetail>;
}

export async function postOverride(
  recordId: string,
  body: { reason: string; corrected_labels?: Record<string, unknown> | null }
): Promise<void> {
  const r = await fetch(`${dashboardApiBase}/studies/${encodeURIComponent(recordId)}/override`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
}

export type GroundingResponse = {
  sentence: string;
  method: string;
  heatmap_png_base64: string;
  image_width: number;
  image_height: number;
};

export async function postGrounding(sentence: string, pngBlob: Blob): Promise<GroundingResponse> {
  const fd = new FormData();
  fd.append("sentence", sentence);
  fd.append("file", pngBlob, "viewport.png");
  const r = await fetch(`${biovilApiBase}/grounding`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<GroundingResponse>;
}

/** Split narrative / impression into clickable phrases (sentence-ish units). */
export function sentencesFromAnalysis(analysis: Record<string, unknown>): string[] {
  const narrative = String(analysis.narrative || "").trim();
  const impression = String(analysis.impression || "").trim();
  const combined = [narrative, impression].filter(Boolean).join("\n\n");
  if (!combined) {
    const summary = String(analysis.screening_summary || "").trim();
    if (summary) return [summary];
    return [];
  }
  const parts = combined
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 2);
  return parts.length ? parts : [combined];
}
