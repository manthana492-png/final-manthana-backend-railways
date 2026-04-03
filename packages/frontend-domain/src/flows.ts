import type { AnalysisResponse, UnifiedAnalysisResult } from "@manthana/api";
import type { ApiClientConfig } from "@manthana/api";
import { createApiClient } from "@manthana/api";

// High-level flow helpers wrapping the shared API client.

export async function runSingleAnalysisFlow(
  file: File,
  modality: string,
  config: ApiClientConfig & {
    patientId?: string;
    clinicalNotes?: string;
    patientContext?: Record<string, unknown>;
    signal?: AbortSignal;
  }
): Promise<AnalysisResponse> {
  const { patientId, clinicalNotes, patientContext, signal, ...clientCfg } = config;
  const client = createApiClient(clientCfg);
  return client.analyzeImage(file, modality, {
    patientId,
    clinicalNotes,
    patientContext,
    signal,
  });
}

export async function runReportFlow(
  analysis: AnalysisResponse,
  language: string,
  clientCfg: ApiClientConfig
) {
  const client = createApiClient(clientCfg);
  return client.generateReport(analysis, language);
}

export async function runUnifiedFlow(
  individualResults: { modality: string; result: AnalysisResponse }[],
  patientId: string,
  language: string,
  clientCfg: ApiClientConfig
): Promise<UnifiedAnalysisResult> {
  const client = createApiClient(clientCfg);
  return client.requestUnifiedReport(individualResults, patientId, language);
}

