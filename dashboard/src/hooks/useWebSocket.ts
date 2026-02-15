"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { useGameState } from "./useGameState";
import { getGameState } from "@/lib/api";
import type { GameInitData, RoundData, GameOverData } from "@/lib/types";

export type WebSocketStatus = "connecting" | "connected" | "reconnecting" | "disconnected" | "idle";

interface UseWebSocketOptions {
  wsBaseUrl?: string;
  gameId?: string | null;
  token?: string | null;
  enabled?: boolean;
}

const MAX_RECONNECT_DELAY = 30000;

export function useWebSocket({
  wsBaseUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765",
  gameId = null,
  token = null,
  enabled = true,
}: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const reconnectDelay = useRef(3000);
  const queuedRounds = useRef<RoundData[]>([]);
  const rafRef = useRef<number | null>(null);
  const [status, setStatus] = useState<WebSocketStatus>("idle");
  const { initGame, pushRound, setGameOver, setStreamingPhase, appendToken, clearStreaming } = useGameState();

  const wsUrl = (() => {
    const base = wsBaseUrl.replace(/\/+$/, "");
    const path = gameId ? `/ws/${gameId}` : "/ws";
    if (token) {
      const separator = path.includes("?") ? "&" : "?";
      return `${base}${path}${separator}token=${encodeURIComponent(token)}`;
    }
    return `${base}${path}`;
  })();

  const cleanup = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    queuedRounds.current = [];
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent reconnect on intentional close
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!enabled) return;
    // Close any existing connection first
    cleanup();

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      setStatus("connecting");

      ws.onopen = () => {
        console.log("[WS] Connected to", wsUrl);
        setStatus("connected");
        reconnectDelay.current = 3000;

        // Fetch cached state via HTTP for late-joining clients
        if (gameId) {
          getGameState(gameId)
            .then((cached) => {
              if (cached.init) {
                console.log("[WS] Replaying cached init + %d rounds", cached.rounds?.length ?? 0);
                initGame(cached.init as GameInitData);
                for (const round of cached.rounds ?? []) {
                  pushRound(round as RoundData);
                }
              }
            })
            .catch(() => {
              // No cached state yet -- will get it via WS when round completes
            });
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "game_init") {
            initGame(data as GameInitData);
          } else if (data.type === "round_update") {
            clearStreaming();
            queuedRounds.current.push(data as RoundData);
            if (rafRef.current === null) {
              rafRef.current = requestAnimationFrame(() => {
                const batch = [...queuedRounds.current];
                queuedRounds.current = [];
                rafRef.current = null;
                for (const round of batch) {
                  pushRound(round);
                }
              });
            }
          } else if (data.type === "game_over") {
            clearStreaming();
            const go = data as GameOverData;
            setGameOver(go.winner, go.final_reflection);
          } else if (data.type === "phase_start") {
            setStreamingPhase(data.phase, data.round);
          } else if (data.type === "token_delta") {
            appendToken(data.agent_id, data.delta);
          } else if (data.type === "phase_complete") {
            // Phase done; streaming tokens stay visible until round_update clears them
          }
        } catch (e) {
          console.error("[WS] Parse error:", e);
        }
      };

      ws.onclose = () => {
        if (!enabled) {
          setStatus("disconnected");
          return;
        }
        const delay = reconnectDelay.current;
        reconnectDelay.current = Math.min(delay * 1.5, MAX_RECONNECT_DELAY);
        console.log(`[WS] Disconnected. Reconnecting in ${Math.round(delay / 1000)}s...`);
        setStatus("reconnecting");
        reconnectTimer.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // onclose will fire after this, which handles reconnect
        setStatus("disconnected");
      };
    } catch {
      // WebSocket constructor can throw on invalid URL
      setStatus("disconnected");
    }
  }, [enabled, wsUrl, gameId, initGame, pushRound, setGameOver, setStreamingPhase, appendToken, clearStreaming, cleanup]);

  useEffect(() => {
    if (!enabled) {
      cleanup();
      setStatus("idle");
      return;
    }
    connect();
    return cleanup;
  }, [connect, enabled, cleanup]);

  return { wsRef, status };
}
