import { orthancBase } from "./config";

/** Orthanc REST study JSON (subset). */
export type OrthancStudy = {
  ID: string;
  MainDicomTags?: Record<string, unknown>;
  Series?: string[];
};

export type OrthancSeries = {
  ID: string;
  Instances?: string[];
};

export async function fetchOrthancStudy(studyId: string): Promise<OrthancStudy> {
  const r = await fetch(`${orthancBase}/studies/${encodeURIComponent(studyId)}`);
  if (!r.ok) throw new Error(`Orthanc study: ${r.status}`);
  return r.json() as Promise<OrthancStudy>;
}

export async function fetchOrthancSeries(seriesId: string): Promise<OrthancSeries> {
  const r = await fetch(`${orthancBase}/series/${encodeURIComponent(seriesId)}`);
  if (!r.ok) throw new Error(`Orthanc series: ${r.status}`);
  return r.json() as Promise<OrthancSeries>;
}

export async function getFirstInstanceId(orthancStudyId: string): Promise<string | null> {
  const study = await fetchOrthancStudy(orthancStudyId);
  const seriesIds = study.Series || [];
  for (const sid of seriesIds) {
    const s = await fetchOrthancSeries(sid);
    const inst = s.Instances?.[0];
    if (inst) return inst;
  }
  return null;
}

export function instanceFileUrl(instanceId: string): string {
  return `${orthancBase}/instances/${encodeURIComponent(instanceId)}/file`;
}
