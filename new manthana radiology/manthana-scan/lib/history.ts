/* ═══ Scan History — in-memory session store ═══ */

import type { AnalysisResponse } from "./types";

export type HistoryStatus = "draft" | "scan_done" | "report_generated";

export interface HistoryEntry {
  id: string; // session id (generated once per page load)
  timestamp: number; // ms since epoch
  modality: string; // e.g. "xray", "mri"
  patientId: string; // e.g. "ANONYMOUS-001"
  imageCount: number;
  findingsCount: number;
  severity: "critical" | "warning" | "info" | "clear" | null;
  impression: string;
  status: HistoryStatus;
  fullResult?: AnalysisResponse;
}

// Module-level store — persists for the browser session, cleared on refresh
let store: HistoryEntry[] = [];

/** Upsert an entry (insert or replace by id). */
export function saveEntry(entry: HistoryEntry): void {
  const idx = store.findIndex((e) => e.id === entry.id);
  if (idx >= 0) {
    store[idx] = entry;
  } else {
    store = [entry, ...store];
  }
}

/** Update fields of an existing entry by id. */
export function patchEntry(id: string, patch: Partial<HistoryEntry>): void {
  store = store.map((e) => (e.id === id ? { ...e, ...patch } : e));
}

/** Return all entries sorted newest first. */
export function getEntries(): HistoryEntry[] {
  return [...store].sort((a, b) => b.timestamp - a.timestamp);
}

/** Get a single entry by id. */
export function getEntry(id: string): HistoryEntry | undefined {
  return store.find((e) => e.id === id);
}

/** Delete a single entry. */
export function deleteEntry(id: string): void {
  store = store.filter((e) => e.id !== id);
}

