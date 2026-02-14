"use client";

import { useEffect, useRef, useCallback } from "react";
import { useGameState } from "./useGameState";
import type { GameInitData, RoundData, GameOverData } from "@/lib/types";

export function useWebSocket(url: string = "ws://localhost:8765") {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const { initGame, pushRound, setGameOver } = useGameState();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[WS] Connected to", url);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "game_init") {
          initGame(data as GameInitData);
        } else if (data.type === "round_update") {
          pushRound(data as RoundData);
        } else if (data.type === "game_over") {
          const go = data as GameOverData;
          setGameOver(go.winner, go.final_reflection);
        }
      } catch (e) {
        console.error("[WS] Failed to parse message:", e);
      }
    };

    ws.onclose = () => {
      console.log("[WS] Disconnected. Reconnecting in 3s...");
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
      ws.close();
    };
  }, [url, initGame, pushRound, setGameOver]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return wsRef;
}
