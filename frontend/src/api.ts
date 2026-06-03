// Typed API client. All requests go through Vite's /api proxy → FastAPI at :8000.

import type {
  ABContextResponse,
  AtBatListResponse,
  GameListResponse,
  PitcherListResponse,
  PitcherProfileResponse,
  QueryRequest,
  QueryResponse,
} from "./types";

const API_BASE = "/api";

export async function listGames(limit = 200): Promise<GameListResponse> {
  const res = await fetch(`${API_BASE}/games?limit=${limit}`);
  if (!res.ok) throw new Error(`/games: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function listAtBats(params?: {
  limit?: number;
  min_pitches?: number;
  max_pitches?: number;
  game_pk?: number;
}): Promise<AtBatListResponse> {
  const q = new URLSearchParams();
  if (params?.limit !== undefined) q.set("limit", String(params.limit));
  if (params?.min_pitches !== undefined) q.set("min_pitches", String(params.min_pitches));
  if (params?.max_pitches !== undefined) q.set("max_pitches", String(params.max_pitches));
  if (params?.game_pk !== undefined) q.set("game_pk", String(params.game_pk));
  const res = await fetch(`${API_BASE}/at-bats?${q}`);
  if (!res.ok) throw new Error(`/at-bats: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getABContext(
  game_pk: number,
  at_bat_number: number,
): Promise<ABContextResponse> {
  const res = await fetch(
    `${API_BASE}/ab-context?game_pk=${game_pk}&at_bat_number=${at_bat_number}`,
  );
  if (!res.ok) throw new Error(`/ab-context: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function postQuery(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`/query: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function listPitchers(
  q: string | null,
  limit = 25,
): Promise<PitcherListResponse> {
  const p = new URLSearchParams();
  if (q && q.trim()) p.set("q", q.trim());
  p.set("limit", String(limit));
  const res = await fetch(`${API_BASE}/pitchers?${p}`);
  if (!res.ok) throw new Error(`/pitchers: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getPitcherProfile(
  pitcherId: number,
  asofDate?: string,
): Promise<PitcherProfileResponse> {
  const p = new URLSearchParams();
  if (asofDate) p.set("asof_date", asofDate);
  const qs = p.toString();
  const url = `${API_BASE}/pitcher/${pitcherId}/profile${qs ? `?${qs}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`/pitcher/profile: ${res.status} ${await res.text()}`);
  return res.json();
}

// ---- MCSim App B (matchup cards) ----

export async function listMcsimPredictions(
  date: string,
): Promise<import("./types").McsimPredictionsForDate> {
  const res = await fetch(`${API_BASE}/mcsim/predictions?date=${encodeURIComponent(date)}`);
  if (!res.ok) throw new Error(`/mcsim/predictions: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getMcsimCard(
  gamePk: number,
  date: string,
): Promise<import("./types").McsimCardResponse> {
  const res = await fetch(
    `${API_BASE}/mcsim/predictions/${gamePk}?date=${encodeURIComponent(date)}`,
  );
  if (!res.ok) throw new Error(`/mcsim/predictions/${gamePk}: ${res.status} ${await res.text()}`);
  return res.json();
}
