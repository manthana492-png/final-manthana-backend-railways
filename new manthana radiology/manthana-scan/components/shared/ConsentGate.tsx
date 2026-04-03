"use client";
import { useState } from "react";

interface Props {
  onAccept: () => void;
  onDecline: () => void;
}

export default function ConsentGate({ onAccept, onDecline }: Props) {
  const [checked, setChecked] = useState(false);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="consent-title"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "oklch(0 0 0 / 0.72)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "var(--space-4)",
      }}
    >
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-xl)",
          padding: "var(--space-8)",
          maxWidth: 480,
          width: "100%",
        }}
      >
        <h2
          id="consent-title"
          style={{
            fontSize: "var(--text-lg)",
            marginBottom: "var(--space-4)",
          }}
        >
          Data Use Consent
        </h2>
        <p
          style={{
            fontSize: "var(--text-sm)",
            color: "var(--color-text-muted)",
            lineHeight: 1.6,
            marginBottom: "var(--space-4)",
          }}
        >
          Manthana Radiologist Copilot will process the uploaded medical image
          using AI models to generate a second-opinion analysis. This tool is
          for decision support only and is not a diagnostic device.
        </p>
        <ul
          style={{
            fontSize: "var(--text-sm)",
            color: "var(--color-text-muted)",
            lineHeight: 1.8,
            paddingLeft: "var(--space-4)",
            marginBottom: "var(--space-6)",
          }}
        >
          <li>Images are processed on-premises and not stored beyond this session.</li>
          <li>No patient-identifiable data is sent to external cloud services.</li>
          <li>All findings require clinical correlation and radiologist verification.</li>
          <li>Use is subject to your institution&apos;s data governance policy.</li>
        </ul>
        <label
          style={{
            display: "flex",
            gap: "var(--space-3)",
            alignItems: "flex-start",
            marginBottom: "var(--space-6)",
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            style={{ marginTop: 3, flexShrink: 0 }}
          />
          <span style={{ fontSize: "var(--text-sm)" }}>
            I confirm that I have obtained appropriate consent for AI-assisted
            analysis of this patient&apos;s data as required under the Digital Personal
            Data Protection Act 2023 (DPDP Act).
          </span>
        </label>
        <div
          style={{
            display: "flex",
            gap: "var(--space-3)",
            justifyContent: "flex-end",
          }}
        >
          <button
            onClick={onDecline}
            style={{
              padding: "var(--space-2) var(--space-6)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-full)",
              fontSize: "var(--text-sm)",
              background: "transparent",
              cursor: "pointer",
              color: "var(--color-text-muted)",
            }}
          >
            Decline
          </button>
          <button
            onClick={onAccept}
            disabled={!checked}
            style={{
              padding: "var(--space-2) var(--space-6)",
              border: "none",
              borderRadius: "var(--radius-full)",
              fontSize: "var(--text-sm)",
              background: checked
                ? "var(--color-primary)"
                : "var(--color-surface-offset)",
              color: checked
                ? "var(--color-text-inverse)"
                : "var(--color-text-faint)",
              cursor: checked ? "pointer" : "not-allowed",
              transition: "background 180ms ease",
            }}
          >
            Accept &amp; Continue
          </button>
        </div>
      </div>
    </div>
  );
}

