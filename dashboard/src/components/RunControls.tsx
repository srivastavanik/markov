"use client";

import { useMemo } from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { ApiGameSummary } from "@/lib/api";

interface RunControlsProps {
  activeGameId: string | null;
  games: ApiGameSummary[];
  loading: boolean;
  error: string | null;
  onRefresh: () => Promise<void>;
  onStart: (mode: "full" | "quick") => Promise<void>;
  onCancel: (gameId: string) => Promise<void>;
}

export function RunControls({
  activeGameId,
  games,
  loading,
  error,
  onRefresh,
  onStart,
  onCancel,
}: RunControlsProps) {
  const running = useMemo(
    () => games.find((g) => g.status === "running" || g.status === "queued"),
    [games],
  );
  const activeSummary = useMemo(
    () => (activeGameId ? games.find((g) => g.game_id === activeGameId) ?? null : null),
    [activeGameId, games],
  );

  return (
    <div className="px-2 py-2 bg-background">
      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
            Run Management
          </span>
          {running ? (
            <Badge variant="outline" className="text-[9px]">Active: {running.game_id}</Badge>
          ) : (
            <Badge variant="outline" className="text-[9px]">Idle</Badge>
          )}
          {activeSummary && (
            <Badge variant="outline" className="text-[9px]">
              {activeSummary.status}
            </Badge>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Button size="sm" className="h-7 text-xs" onClick={() => onStart("full")} disabled={loading || Boolean(running)}>
            Start Game
          </Button>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={onRefresh} disabled={loading}>
            Refresh
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={() => running && onCancel(running.game_id)}
            disabled={!running}
          >
            Cancel
          </Button>
        </div>
      </div>
      {error && <p className="mt-1.5 text-[10px] text-red-600">{error}</p>}
      {activeSummary?.status === "failed" && activeSummary.error && (
        <p className="mt-1.5 text-[10px] text-red-600 truncate">
          Failed: {activeSummary.error}
        </p>
      )}
      <div className="mt-1.5 flex flex-wrap items-center gap-1 text-[10px]">
        <span className="text-muted-foreground">Recent:</span>
        {games.slice(0, 5).map((game) => (
          <Link
            key={game.game_id}
            href={`/replay?gameId=${encodeURIComponent(game.game_id)}`}
            className="rounded border border-black/15 px-1.5 py-0.5 hover:bg-black/5 bg-white truncate max-w-[120px]"
            title={`${game.game_id} · ${game.status}${game.status === "failed" && game.error ? ` · ${game.error}` : ""}`}
          >
            {game.game_id.slice(-8)} · {game.status}
          </Link>
        ))}
        {games.length === 0 && <span className="text-muted-foreground">No runs yet.</span>}
      </div>
    </div>
  );
}

