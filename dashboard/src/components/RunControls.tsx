"use client";

import { useMemo, useState } from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
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
  const [mode, setMode] = useState<"full" | "quick">("quick");
  const running = useMemo(
    () => games.find((g) => g.status === "running" || g.status === "queued"),
    [games],
  );

  return (
    <div className="border-b border-black/10 bg-white px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as "full" | "quick")}
          className="h-9 rounded border border-black/20 px-2 text-sm"
        >
          <option value="quick">Quick Run (10 rounds cap)</option>
          <option value="full">Full Run</option>
        </select>
        <Button size="sm" onClick={() => onStart(mode)} disabled={loading || Boolean(running)}>
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
        {activeGameId && (
          <span className="text-xs text-muted-foreground">
            Live game: <code>{activeGameId}</code>
          </span>
        )}
      </div>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        {games.slice(0, 8).map((game) => (
          <Link
            key={game.game_id}
            href={`/replay?gameId=${encodeURIComponent(game.game_id)}`}
            className="rounded border border-black/15 px-2 py-1 hover:bg-black/5"
          >
            {game.game_id} · {game.status}
          </Link>
        ))}
        {games.length === 0 && <span className="text-muted-foreground">No runs yet.</span>}
      </div>
    </div>
  );
}

