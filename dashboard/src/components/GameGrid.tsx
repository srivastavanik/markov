"use client";

import { useRef, useEffect, useCallback } from "react";
import { useGameState } from "@/hooks/useGameState";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TIER_SIZES, getFamilyColor } from "@/lib/colors";
import type { AgentState, GameEvent } from "@/lib/types";

const CELL_SIZE = 72;
const PADDING = 32;
const LABEL_HEIGHT = 16;

export function GameGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { currentRound, rounds, agents, gridSize } = useGameState();

  const roundData = currentRound > 0 ? rounds[currentRound - 1] : null;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const totalSize = gridSize * CELL_SIZE + PADDING * 2;
    canvas.width = totalSize;
    canvas.height = totalSize;

    // White background
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, totalSize, totalSize);

    // Grid lines
    ctx.strokeStyle = "#E5E7EB";
    ctx.lineWidth = 1;
    for (let i = 0; i <= gridSize; i++) {
      const x = PADDING + i * CELL_SIZE;
      ctx.beginPath();
      ctx.moveTo(x, PADDING);
      ctx.lineTo(x, PADDING + gridSize * CELL_SIZE);
      ctx.stroke();

      const y = PADDING + i * CELL_SIZE;
      ctx.beginPath();
      ctx.moveTo(PADDING, y);
      ctx.lineTo(PADDING + gridSize * CELL_SIZE, y);
      ctx.stroke();
    }

    // Row/col labels
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    for (let i = 0; i < gridSize; i++) {
      ctx.fillText(
        String(i),
        PADDING + i * CELL_SIZE + CELL_SIZE / 2,
        PADDING - 8
      );
      ctx.fillText(
        String(i),
        PADDING - 14,
        PADDING + i * CELL_SIZE + CELL_SIZE / 2 + 4
      );
    }

    // Get agent positions for current round
    const agentList = getAgentPositions(roundData, agents);

    // Adjacency threat lines (between enemy agents)
    for (let i = 0; i < agentList.length; i++) {
      for (let j = i + 1; j < agentList.length; j++) {
        const a = agentList[i];
        const b = agentList[j];
        if (!a.alive || !b.alive) continue;
        if (a.family === b.family) continue;

        const dr = Math.abs(a.position[0] - b.position[0]);
        const dc = Math.abs(a.position[1] - b.position[1]);
        if (dr <= 1 && dc <= 1 && (dr + dc > 0)) {
          const ax = PADDING + a.position[1] * CELL_SIZE + CELL_SIZE / 2;
          const ay = PADDING + a.position[0] * CELL_SIZE + CELL_SIZE / 2;
          const bx = PADDING + b.position[1] * CELL_SIZE + CELL_SIZE / 2;
          const by = PADDING + b.position[0] * CELL_SIZE + CELL_SIZE / 2;

          ctx.strokeStyle = "rgba(220, 38, 38, 0.2)";
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // Draw agents
    for (const agent of agentList) {
      const cx = PADDING + agent.position[1] * CELL_SIZE + CELL_SIZE / 2;
      const cy = PADDING + agent.position[0] * CELL_SIZE + CELL_SIZE / 2;
      const size = TIER_SIZES[agent.tier] || 20;
      const half = size / 2;

      if (!agent.alive) {
        // Ghost outline
        ctx.strokeStyle = `${agent.color}40`;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(cx - half, cy - half, size, size);
      } else {
        // Filled square with border
        ctx.fillStyle = agent.color;
        ctx.fillRect(cx - half, cy - half, size, size);
        ctx.strokeStyle = "#00000020";
        ctx.lineWidth = 1;
        ctx.strokeRect(cx - half, cy - half, size, size);
      }

      // Name label
      ctx.fillStyle = agent.alive ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0.2)";
      ctx.font = `10px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(agent.name, cx, cy + half + LABEL_HEIGHT);
    }

    // Elimination flashes
    if (roundData?.events) {
      for (const ev of roundData.events) {
        if (
          ev.type === "elimination" ||
          ev.type === "mutual_elimination"
        ) {
          const targetId =
            ev.type === "elimination"
              ? (ev.details.target as string)
              : ev.agent_id;
          const target = agentList.find((a) => a.id === targetId);
          if (target) {
            const cx =
              PADDING + target.position[1] * CELL_SIZE + CELL_SIZE / 2;
            const cy =
              PADDING + target.position[0] * CELL_SIZE + CELL_SIZE / 2;
            ctx.fillStyle = "rgba(220, 38, 38, 0.15)";
            ctx.fillRect(
              PADDING + target.position[1] * CELL_SIZE + 1,
              PADDING + target.position[0] * CELL_SIZE + 1,
              CELL_SIZE - 2,
              CELL_SIZE - 2
            );
          }
        }
      }
    }
  }, [currentRound, rounds, agents, gridSize, roundData]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm font-semibold">Grid</CardTitle>
      </CardHeader>
      <CardContent className="flex items-center justify-center p-2">
        <canvas
          ref={canvasRef}
          className="max-w-full"
          style={{ imageRendering: "crisp-edges" }}
        />
      </CardContent>
    </Card>
  );
}

function getAgentPositions(
  roundData: ReturnType<typeof useGameState.getState>["rounds"][0] | null,
  agents: Record<string, AgentState>
): AgentState[] {
  if (roundData?.grid?.agents && roundData.grid.agents.length > 0) {
    return roundData.grid.agents;
  }
  return Object.values(agents);
}
