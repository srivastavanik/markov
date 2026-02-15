"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelGame,
  getApiConfig,
  listGames,
  startGame,
  toWebSocketUrl,
  type ApiGameSummary,
} from "@/lib/api";
import { useGameState } from "@/hooks/useGameState";

export function useGameController() {
  const { activeGameId, setActiveGameId } = useGameState();
  const [wsBaseUrl, setWsBaseUrl] = useState<string>(
    toWebSocketUrl(process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765")
  );
  const [wsToken] = useState<string | null>(
    process.env.NEXT_PUBLIC_WS_TOKEN || null
  );
  const [games, setGames] = useState<ApiGameSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiUp, setApiUp] = useState(true);
  const failCount = useRef(0);

  const refreshGames = useCallback(async () => {
    try {
      const nextGames = await listGames();
      setGames(nextGames);
      setApiUp(true);
      failCount.current = 0;

      // Auto-attach to running game
      const running = nextGames.find(
        (g) => g.status === "running" || g.status === "queued"
      );
      if (running) {
        setActiveGameId(running.game_id);
        return;
      }

      if (!activeGameId) return;
      const active = nextGames.find((g) => g.game_id === activeGameId);
      if (!active) {
        setActiveGameId(null);
        return;
      }
      if (active.status === "failed") {
        setError(active.error || "Game failed.");
        setActiveGameId(null);
        return;
      }
      if (active.status === "completed" || active.status === "cancelled") {
        setActiveGameId(null);
      }
    } catch {
      failCount.current++;
      if (failCount.current >= 3) {
        setApiUp(false);
        // Clear stale game to stop WS reconnect spam
        if (activeGameId) setActiveGameId(null);
      }
    }
  }, [activeGameId, setActiveGameId]);

  // Initial load
  useEffect(() => {
    const load = async () => {
      try {
        const cfg = await getApiConfig();
        setWsBaseUrl(toWebSocketUrl(cfg.ws_url));
      } catch {
        // keep defaults
      }
      await refreshGames();
    };
    void load();
  }, [refreshGames]);

  // Polling: 5s when API is up, 15s when down
  useEffect(() => {
    const interval = apiUp ? 5000 : 15000;
    const timer = setInterval(() => void refreshGames(), interval);
    return () => clearInterval(timer);
  }, [refreshGames, apiUp]);

  const handleStart = useCallback(
    async (mode: "full" | "quick") => {
      setLoading(true);
      setError(null);
      try {
        const started = await startGame(mode, false);
        setActiveGameId(started.game_id);
        await refreshGames();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to start game."
        );
      } finally {
        setLoading(false);
      }
    },
    [refreshGames, setActiveGameId]
  );

  const handleCancel = useCallback(
    async (gameId: string) => {
      setLoading(true);
      setError(null);
      try {
        await cancelGame(gameId);
        await refreshGames();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to cancel game."
        );
      } finally {
        setLoading(false);
      }
    },
    [refreshGames]
  );

  return {
    wsBaseUrl,
    wsToken,
    games,
    loading,
    error,
    apiUp,
    refreshGames,
    handleStart,
    handleCancel,
  };
}
