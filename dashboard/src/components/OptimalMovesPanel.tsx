"use client";

import { useMemo } from "react";
import { computeOptimalMoves, type MoveRecommendation } from "@/lib/moveAnalysis";
import { getFamilyColor } from "@/lib/colors";
import type { AgentState, FamilyConfig } from "@/lib/types";

const ACTION_ICON: Record<string, string> = {
  move: "\u2192",      // →
  eliminate: "\u2694",  // ⚔
  stay: "\u25CB",      // ○
};

const ACTION_LABEL: Record<string, string> = {
  move: "move",
  eliminate: "elim",
  stay: "stay",
};

// ---------------------------------------------------------------------------
// Side gutter panel — renders recommendations for a set of families
// ---------------------------------------------------------------------------

export function OptimalMovesSidePanel({
  familyNames,
  allAgents,
  families,
  gridSize,
  side,
}: {
  familyNames: string[];
  allAgents: AgentState[];
  families: FamilyConfig[];
  gridSize: number;
  side: "left" | "right";
}) {
  const recommendations = useMemo(() => {
    if (allAgents.length === 0) return {};
    return computeOptimalMoves(allAgents, gridSize);
  }, [allAgents, gridSize]);

  const alive = allAgents.filter((a) => a.alive);

  const familyGroups = familyNames
    .map((name) => {
      const fam = families.find((f) => f.name === name);
      const agents = alive
        .filter((a) => a.family === name)
        .sort((a, b) => a.tier - b.tier);
      return {
        name,
        color: fam?.color || getFamilyColor(name),
        provider: fam?.provider ?? "",
        agents,
      };
    })
    .filter((g) => g.agents.length > 0);

  if (familyGroups.length === 0) return <div className="w-[180px] shrink-0" />;

  return (
    <div className={`w-[180px] shrink-0 flex flex-col gap-3 overflow-y-auto py-2 ${side === "left" ? "pr-2" : "pl-2"}`}>
      {familyGroups.map((group) => (
        <div key={group.name}>
          {/* Family header */}
          <div className="flex items-center gap-1.5 mb-1 px-1">
            <span className="text-[9px] font-medium text-black/30 uppercase tracking-wider">
              {group.name}
            </span>
          </div>

          {/* Agent recommendations */}
          <div className="space-y-1.5">
            {group.agents.map((agent) => {
              const rec = recommendations[agent.id];
              if (!rec) return null;
              return (
                <RecommendationCard
                  key={agent.id}
                  agent={agent}
                  rec={rec}
                  familyColor={group.color}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Card for a single agent's recommendation
// ---------------------------------------------------------------------------

function RecommendationCard({
  agent,
  rec,
}: {
  agent: AgentState;
  rec: MoveRecommendation;
  familyColor: string;
}) {
  const icon = ACTION_ICON[rec.action] ?? "?";
  const label = ACTION_LABEL[rec.action] ?? rec.action;
  const confidence = Math.round(rec.score * 100);

  return (
    <div className="px-2 py-1.5 border border-black/[0.06] bg-white">
      <div className="flex items-center justify-between mb-0.5">
        <span className="text-[10px] font-medium text-black/70">
          {agent.name}
        </span>
        <span className="text-[9px] text-black/30 tabular-nums">
          {icon} {label} {confidence}%
        </span>
      </div>
      <p className="text-[9px] text-black/40 leading-[1.4]">
        {rec.rationale}
      </p>
    </div>
  );
}
