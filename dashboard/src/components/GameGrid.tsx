"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import { useGameState } from "@/hooks/useGameState";
import type { AgentState } from "@/lib/types";

const MIN_CELL_SIZE = 48;
const MAX_CELL_SIZE = 120;
const HEADER = 20; // space for col labels

const PROVIDER_LOGOS: Record<string, string> = {
  anthropic: "/logos/anthropic.png",
  openai: "/logos/openai.webp",
  google: "/logos/google.png",
  xai: "/logos/xai.png",
};

// Cache loaded images
const imageCache: Record<string, HTMLImageElement> = {};
function getImage(src: string): HTMLImageElement | null {
  if (imageCache[src]?.complete) return imageCache[src];
  if (!imageCache[src]) {
    const img = new Image();
    img.src = src;
    imageCache[src] = img;
  }
  return null;
}

export function GameGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    currentRound,
    rounds,
    agents,
    gridSize,
    showAdjacencyLines,
    showGhostOutlines,
    setSelectedAgent,
  } = useGameState();
  const [cellSize, setCellSize] = useState(72);

  const roundData = currentRound > 0 ? rounds[currentRound - 1] : null;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const recalc = () => {
      const { width, height } = el.getBoundingClientRect();
      const availW = width / gridSize;
      const availH = (height - HEADER) / gridSize;
      const next = Math.max(MIN_CELL_SIZE, Math.min(MAX_CELL_SIZE, Math.floor(Math.min(availW, availH))));
      if (Number.isFinite(next) && next > 0) setCellSize(next);
    };
    recalc();
    const obs = new ResizeObserver(recalc);
    obs.observe(el);
    return () => obs.disconnect();
  }, [gridSize]);

  // Preload all logos and redraw when ready
  useEffect(() => {
    const srcs = Object.values(PROVIDER_LOGOS);
    let loaded = 0;
    for (const src of srcs) {
      const img = new Image();
      img.src = src;
      img.onload = () => {
        imageCache[src] = img;
        loaded++;
        if (loaded === srcs.length) draw();
      };
      imageCache[src] = img;
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const gridW = gridSize * cellSize;
    const gridH = gridSize * cellSize;
    canvas.width = gridW;
    canvas.height = gridH + HEADER;

    ctx.fillStyle = "#FAFAFA";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Column labels
    ctx.fillStyle = "#71717A";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    for (let i = 0; i < gridSize; i++) {
      ctx.fillText(String(i), i * cellSize + cellSize / 2, 14);
    }

    // Grid lines
    ctx.strokeStyle = "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, HEADER);
      ctx.lineTo(i * cellSize, HEADER + gridH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, HEADER + i * cellSize);
      ctx.lineTo(gridW, HEADER + i * cellSize);
      ctx.stroke();
    }

    // Row labels
    ctx.fillStyle = "#71717A";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    for (let i = 0; i < gridSize; i++) {
      ctx.fillText(String(i), 4, HEADER + i * cellSize + cellSize / 2 + 3);
    }

    const agentList = getAgentPositions(roundData, agents, gridSize);

    // Adjacency lines
    if (showAdjacencyLines) {
      for (let i = 0; i < agentList.length; i++) {
        for (let j = i + 1; j < agentList.length; j++) {
          const a = agentList[i], b = agentList[j];
          if (!a.alive || !b.alive || a.family === b.family) continue;
          const dr = Math.abs(a.position[0] - b.position[0]);
          const dc = Math.abs(a.position[1] - b.position[1]);
          if (dr <= 1 && dc <= 1 && dr + dc > 0) {
            ctx.strokeStyle = "rgba(220,38,38,0.15)";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(a.position[1] * cellSize + cellSize / 2, HEADER + a.position[0] * cellSize + cellSize / 2);
            ctx.lineTo(b.position[1] * cellSize + cellSize / 2, HEADER + b.position[0] * cellSize + cellSize / 2);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }

    // Draw agents as logos
    const logoSize = Math.min(cellSize - 12, 40);
    for (const agent of agentList) {
      const cx = agent.position[1] * cellSize + cellSize / 2;
      const cy = HEADER + agent.position[0] * cellSize + cellSize / 2;

      if (!agent.alive) {
        if (showGhostOutlines) {
          ctx.globalAlpha = 0.15;
          ctx.strokeStyle = agent.color;
          ctx.lineWidth = 1;
          ctx.strokeRect(cx - logoSize / 2, cy - logoSize / 2 - 4, logoSize, logoSize);
          ctx.globalAlpha = 1;
          // Name
          ctx.fillStyle = "rgba(0,0,0,0.15)";
          ctx.font = "9px Inter, system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(agent.name, cx, cy + logoSize / 2 + 4);
        }
        continue;
      }

      // Draw provider logo
      const logoSrc = PROVIDER_LOGOS[agent.provider];
      const img = logoSrc ? getImage(logoSrc) : null;
      if (img) {
        ctx.drawImage(img, cx - logoSize / 2, cy - logoSize / 2 - 4, logoSize, logoSize);
      } else {
        // Fallback colored square
        ctx.fillStyle = agent.color;
        ctx.fillRect(cx - logoSize / 2, cy - logoSize / 2 - 4, logoSize, logoSize);
      }

      // Tier indicator ring
      ctx.strokeStyle = agent.color;
      ctx.lineWidth = agent.tier === 1 ? 2.5 : agent.tier === 2 ? 1.5 : 1;
      ctx.strokeRect(cx - logoSize / 2 - 1, cy - logoSize / 2 - 5, logoSize + 2, logoSize + 2);

      // Name below
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.font = "bold 9px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(agent.name, cx, cy + logoSize / 2 + 6);
    }

    // Elimination flashes
    if (roundData?.events) {
      for (const ev of roundData.events) {
        if (ev.type === "elimination" || ev.type === "mutual_elimination") {
          const targetId = ev.type === "elimination" ? (ev.details.target as string) : ev.agent_id;
          const target = agentList.find((a) => a.id === targetId);
          if (target) {
            ctx.fillStyle = "rgba(220,38,38,0.12)";
            ctx.fillRect(
              target.position[1] * cellSize + 1,
              HEADER + target.position[0] * cellSize + 1,
              cellSize - 2,
              cellSize - 2,
            );
          }
        }
      }
    }
  }, [currentRound, rounds, agents, gridSize, roundData, cellSize, showAdjacencyLines, showGhostOutlines]);

  useEffect(() => { draw(); }, [draw]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const col = Math.floor(x / cellSize);
    const row = Math.floor((y - HEADER) / cellSize);
    if (row < 0 || row >= gridSize || col < 0 || col >= gridSize) return;

    const agentList = getAgentPositions(roundData, agents, gridSize);
    const clicked = agentList.find((a) => a.position[0] === row && a.position[1] === col && a.alive);
    if (clicked) setSelectedAgent(clicked.id);
  };

  return (
    <div ref={containerRef} className="h-full w-full flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="max-w-full max-h-full"
        style={{ imageRendering: "crisp-edges" }}
        onClick={handleClick}
      />
    </div>
  );
}

function getAgentPositions(
  roundData: ReturnType<typeof useGameState.getState>["rounds"][0] | null,
  agents: Record<string, AgentState>,
  gridSize: number,
): AgentState[] {
  const source =
    roundData?.grid?.agents && roundData.grid.agents.length > 0
      ? roundData.grid.agents
      : Object.values(agents);

  return source.map((agent) => ({
    ...agent,
    position: normalizePosition(agent.position, gridSize),
  }));
}

function normalizePosition(
  pos: AgentState["position"] | undefined,
  gridSize: number,
): [number, number] {
  if (Array.isArray(pos) && pos.length === 2 && Number.isFinite(pos[0]) && Number.isFinite(pos[1])) {
    return [
      Math.max(0, Math.min(gridSize - 1, Math.floor(pos[0]))),
      Math.max(0, Math.min(gridSize - 1, Math.floor(pos[1]))),
    ];
  }
  return [0, 0];
}
