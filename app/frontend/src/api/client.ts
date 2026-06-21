const API_BASE = "/api/v1";

export interface MessageData {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export async function getMessages(): Promise<MessageData[]> {
  const res = await fetch(`${API_BASE}/chat/messages`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.messages;
}

export async function clearMessages(): Promise<void> {
  await fetch(`${API_BASE}/chat/messages`, { method: "DELETE" });
}

export async function uploadCV(file: File): Promise<{ text_length: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/cv/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function clearCV(): Promise<void> {
  await fetch(`${API_BASE}/cv/clear`, { method: "DELETE" });
}

export async function getCVStatus(): Promise<boolean> {
  const res = await fetch(`${API_BASE}/cv/status`);
  if (!res.ok) return false;
  const data = await res.json();
  return data.has_cv === true;
}

// ─── Stats API ──────────────────────────────────────────────────────────────

export interface StatsAggregates {
  total_queries: number;
  avg_latency_ms: number;
  fallback_rate: number;
  off_topic_rate: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  intent_distribution: Record<string, number>;
  avg_node_latencies_ms: Record<string, number>;
}

export interface QueryStatRow {
  id: string;
  timestamp: string;
  user_query: string;
  intent: string;
  total_latency_ms: number;
  node_timings_ms: Record<string, number>;
  retrieved_count: number;
  graded_count: number;
  is_grounded: boolean;
  was_fallback: boolean;
  was_off_topic: boolean;
  retry_count: number;
  input_tokens: number | null;
  output_tokens: number | null;
}

export async function clearStats(): Promise<void> {
  await fetch(`${API_BASE}/stats/`, { method: "DELETE" });
}

export async function getStatsAggregates(): Promise<StatsAggregates> {
  const res = await fetch(`${API_BASE}/stats/`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getRecentStats(limit = 50): Promise<QueryStatRow[]> {
  const res = await fetch(`${API_BASE}/stats/recent?limit=${limit}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
