import React, { useEffect, useState } from "react";
import {
  getStatsAggregates,
  getRecentStats,
  clearStats,
  StatsAggregates,
  QueryStatRow,
} from "../api/client";
import "./StatsPage.css";

const NODE_ORDER = [
  "intent_classifier",
  "metadata_extractor",
  "query_translator",
  "retriever",
  "relevance_grader",
  "generator",
  "groundedness_grader",
];

const INTENT_COLORS: Record<string, string> = {
  find_matching: "#6366f1",
  role_transition: "#f59e0b",
  compare: "#10b981",
  clarify_vacancy: "#3b82f6",
  followup: "#8b5cf6",
  aggregate: "#ec4899",
  off_topic: "#ef4444",
};

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function fmtPct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function BarChart({
  data,
  maxValue,
  colorFn,
}: {
  data: [string, number][];
  maxValue: number;
  colorFn?: (key: string) => string;
}) {
  return (
    <div className="bar-chart">
      {data.map(([key, value]) => (
        <div key={key} className="bar-chart__row">
          <span className="bar-chart__label">{key}</span>
          <div className="bar-chart__track">
            <div
              className="bar-chart__fill"
              style={{
                width: maxValue > 0 ? `${(value / maxValue) * 100}%` : "0%",
                background: colorFn ? colorFn(key) : "#6366f1",
              }}
            />
          </div>
          <span className="bar-chart__value">
            {colorFn ? value : fmtMs(value)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function StatsPage() {
  const [agg, setAgg] = useState<StatsAggregates | null>(null);
  const [rows, setRows] = useState<QueryStatRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function loadData() {
    setLoading(true);
    setError(null);
    Promise.all([getStatsAggregates(), getRecentStats(50)])
      .then(([a, r]) => {
        setAgg(a);
        setRows(r);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }

  useEffect(loadData, []);

  async function handleClear() {
    if (!confirm("Clear all statistics? This cannot be undone.")) return;
    setClearing(true);
    try {
      await clearStats();
      loadData();
    } finally {
      setClearing(false);
    }
  }

  if (loading) {
    return (
      <div className="stats-page">
        <div className="stats-loading">Loading statistics…</div>
      </div>
    );
  }

  if (error || !agg) {
    return (
      <div className="stats-page">
        <div className="stats-error">
          Failed to load statistics: {error}
        </div>
      </div>
    );
  }

  const intentEntries = Object.entries(agg.intent_distribution).sort(
    ([, a], [, b]) => b - a
  );
  const maxIntent = Math.max(...intentEntries.map(([, v]) => v), 1);

  const nodeEntries = NODE_ORDER.filter(
    (n) => agg.avg_node_latencies_ms[n] !== undefined
  ).map((n) => [n, agg.avg_node_latencies_ms[n]] as [string, number]);
  const maxNode = Math.max(...nodeEntries.map(([, v]) => v), 1);

  return (
    <div className="stats-page">
      <div className="stats-header">
        <h2 className="stats-title">RAG Pipeline Statistics</h2>
        <button
          className="stats-clear-btn"
          onClick={handleClear}
          disabled={clearing || agg.total_queries === 0}
        >
          {clearing ? "Clearing…" : "Clear stats"}
        </button>
      </div>

      {agg.total_queries === 0 ? (
        <p className="stats-empty">No queries recorded yet. Send a message in the chat first.</p>
      ) : (
        <>
          {/* Summary cards */}
          <div className="stats-cards">
            <div className="stats-card">
              <span className="stats-card__label">Total queries</span>
              <span className="stats-card__value">{agg.total_queries}</span>
            </div>
            <div className="stats-card">
              <span className="stats-card__label">Avg latency</span>
              <span className="stats-card__value">{fmtMs(agg.avg_latency_ms)}</span>
            </div>
            <div className="stats-card stats-card--warn">
              <span className="stats-card__label">Fallback rate</span>
              <span className="stats-card__value">{fmtPct(agg.fallback_rate)}</span>
            </div>
            <div className="stats-card stats-card--danger">
              <span className="stats-card__label">Off-topic rate</span>
              <span className="stats-card__value">{fmtPct(agg.off_topic_rate)}</span>
            </div>
            <div className="stats-card">
              <span className="stats-card__label">Avg input tokens</span>
              <span className="stats-card__value">{agg.avg_input_tokens || "—"}</span>
            </div>
            <div className="stats-card">
              <span className="stats-card__label">Avg output tokens</span>
              <span className="stats-card__value">{agg.avg_output_tokens || "—"}</span>
            </div>
          </div>

          <div className="stats-charts">
            {/* Intent distribution */}
            {intentEntries.length > 0 && (
              <div className="stats-chart-block">
                <h3 className="stats-chart-block__title">Intent Distribution</h3>
                <BarChart
                  data={intentEntries}
                  maxValue={maxIntent}
                  colorFn={(k) => INTENT_COLORS[k] ?? "#94a3b8"}
                />
              </div>
            )}

            {/* Per-node avg latency */}
            {nodeEntries.length > 0 && (
              <div className="stats-chart-block">
                <h3 className="stats-chart-block__title">Avg Node Latency</h3>
                <BarChart data={nodeEntries} maxValue={maxNode} />
              </div>
            )}
          </div>

          {/* Recent queries table */}
          <div className="stats-table-block">
            <h3 className="stats-chart-block__title">Recent Queries</h3>
            <div className="stats-table-wrap">
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Query</th>
                    <th>Intent</th>
                    <th>Latency</th>
                    <th>Retrieved</th>
                    <th>Graded</th>
                    <th>Grounded</th>
                    <th>Retries</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id} className={r.was_off_topic ? "row--muted" : ""}>
                      <td className="cell--time">
                        {new Date(r.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="cell--query" title={r.user_query}>
                        {r.user_query.length > 60
                          ? r.user_query.slice(0, 60) + "…"
                          : r.user_query}
                      </td>
                      <td>
                        <span
                          className="intent-badge"
                          style={{
                            background: INTENT_COLORS[r.intent] ?? "#94a3b8",
                          }}
                        >
                          {r.intent}
                        </span>
                      </td>
                      <td>{fmtMs(r.total_latency_ms)}</td>
                      <td>{r.retrieved_count}</td>
                      <td>{r.graded_count}</td>
                      <td>{r.is_grounded ? "✓" : "✗"}</td>
                      <td>{r.retry_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
