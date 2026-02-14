"use client";

import { useCallback, useEffect, useState } from "react";

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { RoundControls } from "@/components/RoundControls";
import { ThoughtStream } from "@/components/ThoughtStream";
import { GameGrid } from "@/components/GameGrid";
import { DeceptionChart } from "@/components/DeceptionChart";
import { KillTimeline } from "@/components/KillTimeline";
import { RelationshipWeb } from "@/components/RelationshipWeb";
import { AgentDetail } from "@/components/AgentDetail";
import { ExportPanel } from "@/components/ExportPanel";
import { FamilyModelPanel } from "@/components/FamilyModelPanel";
import { RunControls } from "@/components/RunControls";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useGameState } from "@/hooks/useGameState";
import {
  cancelGame,
  getApiConfig,
  listGames,
  startGame,
  toWebSocketUrl,
  type ApiGameSummary,
} from "@/lib/api";

export default function DashboardPage() {
  const { selectedAgent, activeGameId, setActiveGameId } = useGameState();
  const [wsBaseUrl, setWsBaseUrl] = useState<string>(toWebSocketUrl(process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765"));
  const [wsToken] = useState<string | null>(process.env.NEXT_PUBLIC_WS_TOKEN || null);
  const [games, setGames] = useState<ApiGameSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { status } = useWebSocket({
    wsBaseUrl,
    gameId: activeGameId,
    token: wsToken,
    enabled: Boolean(activeGameId),
  });

  const refreshGames = useCallback(async () => {
    try {
      const nextGames = await listGames();
      setGames(nextGames);
      const running = nextGames.find((g) => g.status === "running" || g.status === "queued");
      if (running) {
        setActiveGameId(running.game_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch games.");
    }
  }, [setActiveGameId]);

  useEffect(() => {
    const load = async () => {
      try {
        const cfg = await getApiConfig();
        setWsBaseUrl(toWebSocketUrl(cfg.ws_url));
      } catch {
        // Keep fallback URL when API config is unavailable.
      }
      await refreshGames();
    };
    void load();
  }, [refreshGames]);

  const handleStart = useCallback(async (mode: "full" | "quick") => {
    setLoading(true);
    setError(null);
    try {
      const started = await startGame(mode, false);
      setActiveGameId(started.game_id);
      await refreshGames();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start game.");
    } finally {
      setLoading(false);
    }
  }, [refreshGames, setActiveGameId]);

  const handleCancel = useCallback(async (gameId: string) => {
    setLoading(true);
    setError(null);
    try {
      await cancelGame(gameId);
      await refreshGames();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel game.");
    } finally {
      setLoading(false);
    }
  }, [refreshGames]);

  return (
    <div className="h-screen flex flex-col bg-white">
      <RoundControls status={status} />
      <RunControls
        activeGameId={activeGameId}
        games={games}
        loading={loading}
        error={error}
        onRefresh={refreshGames}
        onStart={handleStart}
        onCancel={handleCancel}
      />
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          {/* Left: Thought Stream */}
          <ResizablePanel defaultSize={30} minSize={20}>
            <div className="h-full p-2">
              <ThoughtStream />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Center: Grid + Charts */}
          <ResizablePanel defaultSize={40} minSize={25}>
            <div className="h-full flex flex-col p-2 gap-2">
              <div className="flex-1 min-h-0">
                <GameGrid />
              </div>
              <div className="h-[220px] shrink-0">
                <DeceptionChart />
              </div>
              <div className="h-[60px] shrink-0">
                <KillTimeline />
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right: Relationships or Agent Detail */}
          <ResizablePanel defaultSize={30} minSize={15}>
            <div className="h-full p-2 flex flex-col gap-2">
              <div className="h-[260px] shrink-0">
                <FamilyModelPanel />
              </div>
              <div className="flex-1 min-h-0">
                {selectedAgent ? <AgentDetail /> : <RelationshipWeb />}
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Bottom export bar */}
      <div className="border-t px-4 py-2 flex items-center justify-end bg-white">
        <ExportPanel />
      </div>
    </div>
  );
}
