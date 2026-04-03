import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import * as cornerstone from "cornerstone-core";
import { instanceFileUrl } from "../lib/orthanc";

export type DicomViewerHandle = {
  getViewportPngBlob: () => Promise<Blob | null>;
};

type Props = {
  instanceId: string | null;
  className?: string;
};

const DicomViewer = forwardRef<DicomViewerHandle, Props>(function DicomViewer(
  { instanceId, className = "" },
  ref
) {
  const elRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const imageIdRef = useRef<string | null>(null);

  useImperativeHandle(ref, () => ({
    async getViewportPngBlob() {
      const el = elRef.current;
      if (!el) return null;
      const canvas = el.querySelector("canvas");
      if (!canvas) return null;
      return new Promise((resolve) => {
        canvas.toBlob((b) => resolve(b), "image/png");
      });
    },
  }));

  useEffect(() => {
    const el = elRef.current;
    if (!el || !instanceId) {
      setError(null);
      return;
    }

    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const url = instanceFileUrl(instanceId);
        const res = await fetch(url);
        if (!res.ok) throw new Error(`DICOM fetch ${res.status}`);
        const buf = await res.arrayBuffer();
        if (cancelled) return;
        const blob = new Blob([buf], { type: "application/dicom" });
        const objectUrl = URL.createObjectURL(blob);
        const imageId = `wadouri:${objectUrl}`;
        if (imageIdRef.current) {
          try {
            const old = imageIdRef.current.replace(/^wadouri:/, "");
            URL.revokeObjectURL(old);
          } catch {
            /* ignore */
          }
        }
        imageIdRef.current = imageId;

        cornerstone.enable(el);
        const image = await cornerstone.loadImage(imageId);
        if (cancelled) return;
        cornerstone.displayImage(el, image);
        cornerstone.fitToWindow(el);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void run();

    return () => {
      cancelled = true;
      const el2 = elRef.current;
      if (el2) {
        try {
          cornerstone.disable(el2);
        } catch {
          /* ignore */
        }
      }
      if (imageIdRef.current) {
        try {
          const u = imageIdRef.current.replace(/^wadouri:/, "");
          URL.revokeObjectURL(u);
        } catch {
          /* ignore */
        }
        imageIdRef.current = null;
      }
    };
  }, [instanceId]);

  if (!instanceId) {
    return (
      <div
        className={`cornerstone-viewport flex items-center justify-center text-slate-500 text-sm ${className}`}
      >
        No DICOM instance resolved for this study.
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <div ref={elRef} className="cornerstone-viewport" />
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-sm text-white">
          Loading DICOM…
        </div>
      )}
      {error && (
        <div className="absolute bottom-2 left-2 right-2 rounded bg-red-950/90 px-2 py-1 font-mono text-xs text-red-200">
          {error}
        </div>
      )}
    </div>
  );
});

export default DicomViewer;
