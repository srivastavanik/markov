"use client";

import { useMemo } from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
    <div className="px-3 py-2 border-b border-black/10 bg-background">
      <Card className="border-black/10 shadow-sm">
        <CardContent className="p-3 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase tracking-wider text-muted-foreground">
              Run Management
            </span>
            {running ? (
              <Badge variant="outline" className="text-[10px]">Active: {running.game_id}</Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">Idle</Badge>
            )}
            {activeGameId && (
              <span className="text-xs text-muted-foreground">
                Stream: <code>{activeGameId}</code>
              </span>
            )}
            {activeSummary && (
              <Badge variant="outline" className="text-[10px]">
                Status: {activeSummary.status}
              </Badge>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" onClick={() => onStart("full")} disabled={loading || Boolean(running)}>
              Start Game
            </Button>
            <Button size="sm" variant="outline" onClick={onRefresh} disabled={loading}>
              Refresh
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => running && onCancel(running.game_id)}
              disabled={!running}
            >
              Cancel Active
            </Button>
          </div>
        </CardContent>
      </Card>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      {activeSummary?.status === "failed" && activeSummary.error && (
        <p className="mt-2 text-xs text-red-600">
          Active run failed: {activeSummary.error}
        </p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs px-1">
        <span className="text-muted-foreground mr-1">Recent:</span>
        {games.slice(0, 8).map((game) => (
          <Link
            key={game.game_id}
            href={`/replay?gameId=${encodeURIComponent(game.game_id)}`}
            className="rounded border border-black/15 px-2 py-1 hover:bg-black/5 bg-white"
            title={game.status === "failed" && game.error ? game.error : undefined}
          >
            {game.game_id} · {game.status}
            {game.status === "failed" && game.error ? ` · ${game.error}` : ""}
          </Link>
        ))}
        {games.length === 0 && <span className="text-muted-foreground">No runs yet.</span>}
      </div>
    </div>
  );
}

