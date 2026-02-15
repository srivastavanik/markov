export type ApiGameStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface ApiGameSummary {
  game_id: string;
  status: ApiGameStatus;
  mode: "full" | "quick";
  started_at: string;
  ended_at: string | null;
  output_dir: string | null;
  error: string | null;
  winner: string | null;
  total_rounds: number | null;
}

export interface ApiConfig {
  ws_url: string;
  ws_token_required: boolean;
}

export interface ApiSeriesSummary {
  series_id: string;
  series_type: string;
  provider: string | null;
  num_games: number;
  created_at: string;
  aggregate_metrics: Record<string, unknown> | null;
}

export interface ApiSeriesDetail {
  series_id: string;
  series_type: string;
  provider: string | null;
  num_games: number;
  config: Record<string, unknown>;
  games: ApiGameSummary[];
  aggregate_metrics: Record<string, unknown> | null;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}

export async function getApiConfig(): Promise<ApiConfig> {
  return request<ApiConfig>("/api/config");
}

export async function startGame(mode: "full" | "quick", verbose: boolean): Promise<{ game_id: string }> {
  return request<{ game_id: string }>("/api/games", {
    method: "POST",
    body: JSON.stringify({ mode, verbose }),
  });
}

export async function cancelGame(gameId: string): Promise<{ game_id: string; status: ApiGameStatus }> {
  return request<{ game_id: string; status: ApiGameStatus }>(`/api/games/${gameId}/cancel`, {
    method: "POST",
  });
}

export async function listGames(): Promise<ApiGameSummary[]> {
  return request<ApiGameSummary[]>("/api/games");
}

export async function getGameState(gameId: string): Promise<{ init: unknown; rounds: unknown[] }> {
  return request<{ init: unknown; rounds: unknown[] }>(`/api/games/${gameId}/state`);
}

export async function getReplay(gameId: string): Promise<unknown> {
  return request<unknown>(`/api/games/${gameId}/replay`);
}

export async function getGameMetrics(gameId: string): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`/api/games/${gameId}/metrics`);
}

export async function getGameAnalysis(gameId: string): Promise<unknown> {
  return request<unknown>(`/api/games/${gameId}/analysis`);
}

export async function listSeries(): Promise<ApiSeriesSummary[]> {
  return request<ApiSeriesSummary[]>("/api/series");
}

export async function getSeries(seriesId: string): Promise<ApiSeriesDetail> {
  return request<ApiSeriesDetail>(`/api/series/${seriesId}`);
}

export function toWebSocketUrl(httpUrl: string): string {
  if (httpUrl.startsWith("https://")) return `wss://${httpUrl.slice("https://".length)}`;
  if (httpUrl.startsWith("http://")) return `ws://${httpUrl.slice("http://".length)}`;
  return httpUrl;
}

