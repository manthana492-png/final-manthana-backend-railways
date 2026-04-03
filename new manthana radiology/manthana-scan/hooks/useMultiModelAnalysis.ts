"use client";
import { useState, useCallback, useRef } from "react";
import type {
  MultiModelSession,
  MultiModelStage,
  MultiModelUpload,
  MultiModelResult,
  UnifiedAnalysisResult,
  AnalysisResponse,
} from "@/lib/types";
import { analyzeImage, requestUnifiedReport } from "@/lib/api";
import { AnalysisCancelledError } from "@/lib/errors";

const INITIAL_SESSION: MultiModelSession = {
  id: "",
  selectedModalities: [],
  uploads: [],
  copilotActivated: false,
  individualResults: [],
  unifiedResult: null,
  stage: "selecting",
  currentProcessingIndex: -1,
};

export function useMultiModelAnalysis() {
  const [session, setSession] = useState<MultiModelSession>(INITIAL_SESSION);
  const abortRef = useRef<AbortController | null>(null);

  /* ── Start a new multi-model session ── */
  const startSession = useCallback(() => {
    setSession({
      ...INITIAL_SESSION,
      id: crypto.randomUUID(),
      stage: "selecting",
    });
  }, []);

  /* ── Toggle modality selection (max 4) ── */
  const toggleModality = useCallback((modalityId: string) => {
    setSession((s) => {
      const isSelected = s.selectedModalities.includes(modalityId);
      let updated: string[];
      if (isSelected) {
        updated = s.selectedModalities.filter((m) => m !== modalityId);
      } else {
        if (s.selectedModalities.length >= 4) return s; // max 4
        updated = [...s.selectedModalities, modalityId];
      }
      return { ...s, selectedModalities: updated };
    });
  }, []);

  /* ── Confirm modality selection → move to uploading ── */
  const confirmSelection = useCallback(() => {
    setSession((s) => ({
      ...s,
      stage: "uploading",
      uploads: s.selectedModalities.map((m) => ({
        modality: m,
        files: [],
        urls: [],
        uploaded: false,
      })),
    }));
  }, []);

  /* ── Set files for a specific modality upload step ── */
  const setUploadFiles = useCallback((modalityId: string, files: File[]) => {
    setSession((s) => ({
      ...s,
      uploads: s.uploads.map((u) =>
        u.modality === modalityId
          ? {
              ...u,
              files,
              urls: files.map((f) => URL.createObjectURL(f)),
              uploaded: files.length > 0,
            }
          : u
      ),
    }));
  }, []);

  /* ── All uploads done → show copilot activation ── */
  const proceedToConfirm = useCallback(() => {
    setSession((s) => ({ ...s, stage: "confirming" }));
  }, []);

  /* ── Activate Copilot → start sequential processing ── */
  const activateCopilot = useCallback(
    async (patientId: string) => {
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      setSession((s) => ({
        ...s,
        copilotActivated: true,
        stage: "processing",
        currentProcessingIndex: 0,
        individualResults: [],
      }));

      const results: MultiModelResult[] = [];

      // Get current uploads from state
      const currentSession = await new Promise<MultiModelSession>((resolve) => {
        setSession((s) => {
          resolve(s);
          return s;
        });
      });

      for (let i = 0; i < currentSession.uploads.length; i++) {
        if (ctrl.signal.aborted) return;

        const upload = currentSession.uploads[i];
        setSession((s) => ({ ...s, currentProcessingIndex: i }));

        try {
          // Use the first file for each modality (primary scan)
          const file = upload.files[0];
          if (!file) continue;

          const result = await analyzeImage(
            file,
            upload.modality,
            undefined,
            undefined,
            undefined,
            ctrl.signal
          );

          if (ctrl.signal.aborted) return;

          results.push({ modality: upload.modality, result });
          setSession((s) => ({
            ...s,
            individualResults: [...results],
          }));
        } catch (err: unknown) {
          if (err instanceof AnalysisCancelledError || ctrl.signal.aborted) {
            return;
          }
          const message =
            err instanceof Error
              ? err.message
              : `${upload.modality} analysis failed.`;
          // Return instead of just breaking: with incomplete results a unified
          // report would be misleading, so we abort the whole activation.
          setSession((s) => ({
            ...s,
            stage: "error",
            errorMessage: message,
          }));
          return;
        }
      }

      // All individual analyses done → request unified report
      if (ctrl.signal.aborted) return;

      setSession((s) => ({ ...s, stage: "unifying" }));

      // NOTE: unified report endpoint (/unified-report) may not be live yet.
      // When it fails, session goes to stage: "error" with a real error message.
      // This is intentional — no mock fallback. If tests need a fake unified result
      // before the backend ships, reintroduce a mock here in a controlled way.
      try {
        const unifiedResult = await requestUnifiedReport(results, patientId);
        if (ctrl.signal.aborted) return;

        setSession((s) => ({
          ...s,
          unifiedResult,
          stage: "complete",
        }));
      } catch (err: unknown) {
        if (ctrl.signal.aborted) return;
        const message =
          err instanceof Error
            ? err.message
            : "Unified report generation failed.";
        setSession((s) => ({
          ...s,
          stage: "error",
          errorMessage: message,
        }));
      }
    },
    []
  );

  /* ── Go back to a previous stage ── */
  const goBack = useCallback(() => {
    setSession((s) => {
      switch (s.stage) {
        case "uploading":
          return { ...s, stage: "selecting" as MultiModelStage };
        case "confirming":
          return { ...s, stage: "uploading" as MultiModelStage };
        default:
          return s;
      }
    });
  }, []);

  /* ── Reset everything ── */
  const resetMultiModel = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    // Revoke all object URLs
    session.uploads.forEach((u) => u.urls.forEach((url) => URL.revokeObjectURL(url)));
    setSession(INITIAL_SESSION);
  }, [session.uploads]);

  return {
    session,
    startSession,
    toggleModality,
    confirmSelection,
    setUploadFiles,
    proceedToConfirm,
    activateCopilot,
    goBack,
    resetMultiModel,
  };
}
