// Shapes of core's REST/SSE payloads (see docs/core.md). All data is SYNTHETIC.

export type Reading = {
  ts: string;
  zone: string;
  temp_c: number | null; // null on sensor dropout
  compressor_on: boolean;
  compressor_current_a: number;
};

export type AlertKind = "anomaly" | "breach_risk" | "breached";

export type Alert = {
  id: number | null;
  zone: string;
  kind: AlertKind;
  severity: string;
  opened_at: string;
  closed_at: string | null;
  peak_score: number | null;
  summary: string | null;
  model_version: string | null;
};

export type Incident = {
  id: number | null;
  received_at: string;
  zone: string | null;
  text: string;
  label: "damage" | "temperature" | "delay" | "safety";
  confidence: number; // top softmax score, a ranking signal and NOT a calibrated probability
  needs_review: boolean;
  model_version: string | null;
};

export type ZoneSummary = {
  zone: string;
  latest: Reading | null;
};

export type StreamEvent =
  | { type: "reading"; reading: Reading }
  | { type: "alert"; action: string; alert: Alert }
  | { type: "incident"; incident: Incident };

export type DriftRow = {
  zone: string;
  feature: string;
  psi: number | null;
  n: number;
  state: string;
};

export type ScoreRow = {
  model: string;
  psi: number | null;
  n_baseline: number;
  n_recent: number;
  state: string;
};

export type DriftReport = {
  checked_at: number | null;
  thresholds?: { warn: number; drift: number };
  inputs: DriftRow[];
  predictions: ScoreRow[];
  errors: string[];
};

export const LIMIT_C = -18; // the breach limit used by the models
