/** Pipeline may store relative heatmap paths served by the gateway. */
export function resolveHeatmapUrl(raw: string | undefined | null): string | null {
  if (!raw || typeof raw !== "string") return null;
  if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
  const base = (import.meta.env.VITE_GATEWAY_PUBLIC_BASE || "").replace(/\/$/, "");
  if (raw.startsWith("/") && base) return `${base}${raw}`;
  return raw.startsWith("/") ? null : raw;
}
