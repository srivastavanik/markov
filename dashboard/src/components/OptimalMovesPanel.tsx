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
// Corner panel — renders recommendations for a single family
// ---------------------------------------------------------------------------

export function OptimalMovesCornerPanel({
  familyName,
  allAgents,
  families,
  gridSize,
}: {
  familyName: string;
  allAgents: AgentState[];
  families: FamilyConfig[];
  gridSize: number;
}) {
  const recommendations = useMemo(() => {
    if (allAgents.length === 0) return {};
    return computeOptimalMoves(allAgents, gridSize);
  }, [allAgents, gridSize]);

  const fam = families.find((f) => f.name === familyName);
  const alive = allAgents
    .filter((a) => a.alive && a.family === familyName)
    .sort((a, b) => a.tier - b.tier);

  if (alive.length === 0) return null;

  const color = fam?.color || getFamilyColor(familyName);

  return (
    <div className="flex flex-col gap-2 overflow-y-auto p-2">
      {/* Family header */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] font-semibold text-black/40 uppercase tracking-wider">
          {familyName}
        </span>
      </div>

      {/* Agent recommendations */}
      {alive.map((agent) => {
        const rec = recommendations[agent.id];
        if (!rec) return null;
        return (
          <RecommendationCard
            key={agent.id}
            agent={agent}
            rec={rec}
            familyColor={color}
          />
        );
      })}
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
    <div className="px-2.5 py-2 border border-black/[0.06] bg-white">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-semibold text-black/70">
          {agent.name}
        </span>
        <span className="text-[10px] text-black/35 tabular-nums font-medium">
          {icon} {label} {confidence}%
        </span>
      </div>
      <p className="text-[10px] text-black/45 leading-[1.5]">
        {rec.rationale}
      </p>
    </div>
  );
}
