"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGameState } from "@/hooks/useGameState";
import { getFamilyColor } from "@/lib/colors";

interface KillEvent {
  round: number;
  type: "elimination" | "mutual_elimination";
  attacker: string;
  target: string;
  targetFamily: string;
  attackerFamily: string;
}

export function KillTimeline() {
  const { rounds, currentRound, agents } = useGameState();
  const visibleRounds = rounds.slice(0, currentRound);

  const kills = useMemo(() => {
    const events: KillEvent[] = [];
    for (const round of visibleRounds) {
      for (const ev of round.events || []) {
        if (ev.type === "elimination") {
          const targetId = ev.details.target as string;
          const target = agents[targetId];
          const attacker = agents[ev.agent_id];
          events.push({
            round: round.round,
            type: "elimination",
            attacker: attacker?.name || ev.agent_id,
            target: target?.name || targetId,
            targetFamily: target?.family || "",
            attackerFamily: attacker?.family || "",
          });
        } else if (ev.type === "mutual_elimination") {
          const targetId = ev.details.target as string;
          const target = agents[targetId];
          const agent = agents[ev.agent_id];
          events.push({
            round: round.round,
            type: "mutual_elimination",
            attacker: agent?.name || ev.agent_id,
            target: target?.name || targetId,
            targetFamily: target?.family || "",
            attackerFamily: agent?.family || "",
          });
        }
      }
    }
    return events;
  }, [visibleRounds, agents]);

  const maxRound = Math.max(currentRound, 1);

  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm font-semibold">Elimination Timeline</CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-3">
        {kills.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No eliminations yet
          </p>
        ) : (
          <div className="relative h-10">
            {/* Timeline bar */}
            <div className="absolute top-1/2 left-0 right-0 h-px bg-gray-200 -translate-y-1/2" />

            {/* Round markers */}
            {Array.from({ length: maxRound }, (_, i) => i + 1).map((r) => (
              <div
                key={r}
                className="absolute top-1/2 -translate-y-1/2 w-px h-2 bg-gray-300"
                style={{ left: `${(r / maxRound) * 100}%` }}
              />
            ))}

            {/* Kill markers */}
            {kills.map((kill, i) => {
              const leftPct = (kill.round / maxRound) * 100;
              const color = getFamilyColor(kill.targetFamily);

              return (
                <Tooltip key={i}>
                  <TooltipTrigger asChild>
                    <div
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 cursor-pointer"
                      style={{ left: `${leftPct}%` }}
                    >
                      {kill.type === "mutual_elimination" ? (
                        <div className="flex">
                          <div
                            className="w-3 h-3 rounded-full border-2 border-white"
                            style={{ backgroundColor: getFamilyColor(kill.attackerFamily) }}
                          />
                          <div
                            className="w-3 h-3 rounded-full border-2 border-white -ml-1.5"
                            style={{ backgroundColor: color }}
                          />
                        </div>
                      ) : (
                        <div
                          className="w-3.5 h-3.5 rounded-full border-2 border-white shadow-sm"
                          style={{ backgroundColor: color }}
                        />
                      )}
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="top">
                    <p className="text-xs font-medium">
                      R{kill.round}: {kill.attacker}{" "}
                      {kill.type === "mutual_elimination" ? "<->" : "->"}{" "}
                      {kill.target}
                    </p>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
