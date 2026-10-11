"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import { fetchAlerts, fetchDrift, fetchReadings, fetchZones, STREAM_URL } from "./api";
import { applyEvents, empty, seed, type State } from "./state";
import type { DriftReport, StreamEvent } from "./types";

const FLUSH_MS = 500; // the simulator can replay at 60x: batch events instead of rendering each one
const HISTORY_MIN = 360;

type Action = { t: "events"; events: StreamEvent[] } | { t: "seed"; history: Parameters<typeof seed>[1]; alerts: Parameters<typeof seed>[2] };

function reduce(state: State, a: Action): State {
  return a.t === "events" ? applyEvents(state, a.events) : seed(state, a.history, a.alerts);
}

export type Connection = "connecting" | "live" | "reconnecting" | "offline";

export function useColdChain() {
  const [state, dispatch] = useReducer(reduce, empty);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [driftError, setDriftError] = useState<string | null>(null);
  const buffer = useRef<StreamEvent[]>([]);

  // live stream first, so nothing is missed while the history loads; duplicates are dropped by the reducer
  useEffect(() => {
    const es = new EventSource(STREAM_URL);
    const parse = (m: MessageEvent) => {
      try {
        buffer.current.push(JSON.parse(m.data) as StreamEvent);
      } catch {
        /* a malformed event is skipped, not fatal */
      }
    };
    es.addEventListener("reading", parse);
    es.addEventListener("alert", parse);
    es.onopen = () => setConnection("live");
    es.onerror = () => setConnection(es.readyState === EventSource.CLOSED ? "offline" : "reconnecting");
    const timer = setInterval(() => {
      if (buffer.current.length) {
        const events = buffer.current;
        buffer.current = [];
        dispatch({ t: "events", events });
      }
    }, FLUSH_MS);
    return () => {
      clearInterval(timer);
      es.close();
    };
  }, []);

  // history (and again whenever the stream reconnects, to fill any gap)
  useEffect(() => {
    if (connection !== "live") return;
    let cancelled = false;
    (async () => {
      try {
        const zones = await fetchZones();
        const [alerts, ...rows] = await Promise.all([
          fetchAlerts(),
          ...zones.map((z) => fetchReadings(z.zone, HISTORY_MIN)),
        ]);
        if (cancelled) return;
        dispatch({ t: "seed", history: Object.fromEntries(zones.map((z, i) => [z.zone, rows[i]])), alerts });
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [connection]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const r = await fetchDrift();
        if (!cancelled) {
          setDrift(r);
          setDriftError(null);
        }
      } catch (e) {
        if (!cancelled) setDriftError(e instanceof Error ? e.message : String(e));
      }
    };
    load();
    const t = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  return { state, connection, error, drift, driftError };
}
