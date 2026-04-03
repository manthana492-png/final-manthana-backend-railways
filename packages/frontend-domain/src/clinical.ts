// Shared clinical notes and patient context builders used by both frontends.

export function buildClinicalNotesForApi(ctx: {
  age?: string;
  gender?: string;
  location?: string;
  tobaccoUse?: string;
  fastingStatus?: string;
  medications?: string;
}): string {
  const parts: string[] = [];
  if (ctx.tobaccoUse) parts.push(`tobacco_use:${ctx.tobaccoUse}`);
  if (ctx.age) parts.push(`age:${ctx.age}`);
  if (ctx.gender) parts.push(`gender:${ctx.gender}`);
  if (ctx.location) parts.push(`location:${ctx.location}`);
  if (ctx.fastingStatus && ctx.fastingStatus !== "unknown") {
    parts.push(`fasting:${ctx.fastingStatus}`);
  }
  if (ctx.medications?.trim()) parts.push(`medications:${ctx.medications.trim()}`);
  return parts.join("; ");
}

export function buildPatientContextJsonForApi(ctx: {
  age?: string;
  gender?: string;
  location?: string;
  tobaccoUse?: string;
}): Record<string, unknown> | undefined {
  const out: Record<string, unknown> = {};
  const ageStr = ctx.age?.trim();
  if (ageStr) {
    const n = parseInt(ageStr, 10);
    if (Number.isFinite(n)) out.age = n;
  }
  if (ctx.gender?.trim()) {
    const g = ctx.gender.trim().toUpperCase();
    out.sex = g === "M" || g === "F" ? g : "Unknown";
  }
  if (ctx.location?.trim()) out.location_body = ctx.location.trim();
  if (ctx.tobaccoUse?.trim()) out.history = ctx.tobaccoUse.trim();
  return Object.keys(out).length > 0 ? out : undefined;
}

