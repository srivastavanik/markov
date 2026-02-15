"use client";

import { AppTopBar } from "@/components/AppTopBar";
import { RunControls } from "@/components/RunControls";
import { LiveWorkspace } from "@/components/layout/LiveWorkspace";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useGameController } from "@/hooks/useGameController";
import { useGameState } from "@/hooks/useGameState";

export default function DashboardPage() {
  const { activeGameId, setActiveGameId } = useGameState();
  const {
    wsBaseUrl,
    wsToken,
    games,
    loading,
    error,
    apiUp,
    refreshGames,
    handleStart,
    handleCancel,
  } = useGameController();
  useKeyboardShortcuts();

  // Only enable WS when there's an active game AND the API is reachable
  const wsEnabled = Boolean(activeGameId) && apiUp;

  const { status } = useWebSocket({
    wsBaseUrl,
    gameId: activeGameId,
    token: wsToken,
    enabled: wsEnabled,
  });

  const displayStatus = !apiUp
    ? "disconnected"
    : !activeGameId
      ? "idle"
      : status;

  return (
    <div className="h-screen flex flex-col bg-background">
      <AppTopBar
        status={displayStatus}
        games={games}
        onSelectGame={setActiveGameId}
      />
      <LiveWorkspace
        runControls={
          <RunControls
            activeGameId={activeGameId}
            games={games}
            loading={loading}
            error={error}
            onRefresh={refreshGames}
            onStart={handleStart}
            onCancel={handleCancel}
          />
        }
      />
    </div>
  );
}
