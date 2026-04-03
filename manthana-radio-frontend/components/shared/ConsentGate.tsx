"use client";

import { useState } from "react";
import { GATEWAY_URL } from "@/lib/constants";
import { getGatewayAuthToken } from "@/lib/auth-token";

interface Props {
  onAccept: (patientId: string) => void;
}

export function ConsentGate({ onAccept }: Props) {
  const [patientId, setPatientId] = useState("");
  const [accepted, setAccepted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAccept = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const token = getGatewayAuthToken();
      const res = await fetch(`${GATEWAY_URL}/consent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          patient_id: patientId || "ANONYMOUS",
          purpose: "radiology_second_opinion",
          informed_by: "clinician",
        }),
      });
      if (!res.ok) {
        throw new Error(`Consent failed (${res.status})`);
      }
      onAccept(patientId || "ANONYMOUS");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to record consent. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="consent-backdrop"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 40,
        background: "radial-gradient(circle at top, rgba(0,0,0,0.75), rgba(0,0,0,0.9))",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        className="glass-panel"
        style={{
          maxWidth: 520,
          width: "90%",
          padding: "20px 22px 18px",
          borderRadius: "var(--r-lg)",
          border: "1px solid rgba(255,255,255,0.08)",
          boxShadow: "0 24px 80px rgba(0,0,0,0.65)",
        }}
      >
        <p
          className="font-display"
          style={{
            fontSize: 13,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--gold-300)",
            marginBottom: 10,
          }}
        >
          Patient Data Consent — DPDP Act 2023
        </p>
        <p className="font-body" style={{ fontSize: 11, color: "var(--text-60)", marginBottom: 12, lineHeight: 1.6 }}>
          This AI tool processes medical images and related clinical information to provide a{" "}
          <strong>second-opinion radiology analysis</strong>. Data is processed within the Manthana
          environment and is intended for use by qualified clinicians only. All findings are{" "}
          <strong>advisory</strong> and must be reviewed and confirmed by a radiologist before any
          clinical decisions are made.
        </p>
        <p className="font-body" style={{ fontSize: 11, color: "var(--text-45)", marginBottom: 12 }}>
          By proceeding, you confirm that patient consent for this specific AI-based analysis has
          been obtained in accordance with the{" "}
          <strong>Digital Personal Data Protection (DPDP) Act 2023</strong> and your institution&apos;s
          policies.
        </p>

        <div style={{ marginBottom: 10 }}>
          <label className="text-caption" style={{ fontSize: 10, color: "var(--text-40)", display: "block", marginBottom: 4 }}>
            Patient ID (optional)
          </label>
          <input
            type="text"
            placeholder="e.g. HOSP-123456 (leave blank for anonymous)"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 10px",
              borderRadius: 6,
              border: "1px solid var(--glass-border)",
              background: "rgba(10,10,10,0.85)",
              color: "var(--text-80)",
              fontSize: 11,
              fontFamily: "var(--font-body)",
            }}
          />
        </div>

        <label
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 8,
            fontSize: 11,
            color: "var(--text-65)",
            marginBottom: 10,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={accepted}
            onChange={(e) => setAccepted(e.target.checked)}
            style={{ marginTop: 2 }}
          />
          <span>
            I confirm that <strong>valid consent</strong> has been obtained from the patient (or
            authorised representative) for processing their medical data using this AI system, in
            line with the DPDP Act 2023.
          </span>
        </label>

        {error && (
          <p className="text-caption" style={{ fontSize: 10, color: "var(--danger-300)", marginBottom: 8 }}>
            {error}
          </p>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 6 }}>
          <button
            type="button"
            disabled={!accepted || submitting}
            onClick={handleAccept}
            className="btn-teal"
            style={{
              fontSize: 11,
              padding: "7px 18px",
              opacity: !accepted || submitting ? 0.6 : 1,
              cursor: !accepted || submitting ? "not-allowed" : "pointer",
            }}
          >
            {submitting ? "Recording consent…" : "Proceed to analysis"}
          </button>
        </div>
      </div>
    </div>
  );
}

