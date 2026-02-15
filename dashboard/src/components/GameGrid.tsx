"use client";

import { useRef, useEffect, useCallback, useState, useMemo } from "react";
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

const TIER_BORDER: Record<number, string> = { 1: "2.5px", 2: "1.5px", 3: "1px" };

// Arrow colors
const ARROW_MOVE = "rgba(59,130,246,0.55)";   // blue — empty cell movement
const ARROW_ATTACK = "rgba(220,38,38,0.65)";  // red — enemy agent (attack)
const ARROW_ALLY = "rgba(34,197,94,0.35)";     // green — same-family agent

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
  const [hoveredAgent, setHoveredAgent] = useState<string | null>(null);

  const roundData = currentRound > 0 ? rounds[currentRound - 1] : null;

  const agentList = useMemo(
    () => getAgentPositions(roundData, agents, gridSize),
    [roundData, agents, gridSize],
  );

  const gridW = gridSize * cellSize;
  const gridH = gridSize * cellSize;
  const canvasH = gridH + HEADER;
  const logoSize = Math.min(cellSize - 12, 40);

  // Build occupancy map for hover arrows
  const occupancyMap = useMemo(() => {
    const map: Record<string, AgentState> = {};
    for (const a of agentList) {
      if (a.alive) map[`${a.position[0]},${a.position[1]}`] = a;
    }
    return map;
  }, [agentList]);

  // Compute hover arrows
  const hoverArrows = useMemo(() => {
    if (!hoveredAgent) return [];
    const agent = agentList.find((a) => a.id === hoveredAgent);
    if (!agent || !agent.alive) return [];

    const [row, col] = agent.position;
    const arrows: { toRow: number; toCol: number; color: string; type: "move" | "attack" | "ally" }[] = [];

    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const nr = row + dr;
        const nc = col + dc;
        if (nr < 0 || nr >= gridSize || nc < 0 || nc >= gridSize) continue;

        const occupant = occupancyMap[`${nr},${nc}`];
        if (!occupant) {
          arrows.push({ toRow: nr, toCol: nc, color: ARROW_MOVE, type: "move" });
        } else if (occupant.family !== agent.family) {
          arrows.push({ toRow: nr, toCol: nc, color: ARROW_ATTACK, type: "attack" });
        } else {
          arrows.push({ toRow: nr, toCol: nc, color: ARROW_ALLY, type: "ally" });
        }
      }
    }
    return arrows;
  }, [hoveredAgent, agentList, occupancyMap, gridSize]);

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

  // Canvas: grid lines, labels, adjacency lines, elimination flashes
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = gridW;
    canvas.height = canvasH;

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

    // Adjacency lines (only when not hovering)
    if (showAdjacencyLines && !hoveredAgent) {
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
  }, [gridSize, cellSize, gridW, gridH, canvasH, agentList, roundData, showAdjacencyLines, hoveredAgent]);

  useEffect(() => { draw(); }, [draw]);

  const hoveredAgentData = hoveredAgent ? agentList.find((a) => a.id === hoveredAgent) : null;

  return (
    <div ref={containerRef} className="h-full w-full flex items-center justify-center">
      <div
        className="relative"
        style={{ width: gridW, height: canvasH }}
        onMouseLeave={() => setHoveredAgent(null)}
      >
        <canvas
          ref={canvasRef}
          className="absolute inset-0"
          style={{ imageRendering: "crisp-edges" }}
        />

        {/* SVG overlay for hover arrows */}
        {hoveredAgent && hoveredAgentData && (
          <svg
            className="absolute inset-0 pointer-events-none"
            width={gridW}
            height={canvasH}
            style={{ zIndex: 5 }}
          >
            <defs>
              <marker id="arrow-move" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill={ARROW_MOVE} />
              </marker>
              <marker id="arrow-attack" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                <path d="M0,0 L7,3.5 L0,7 Z" fill={ARROW_ATTACK} />
              </marker>
              <marker id="arrow-ally" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
                <path d="M0,0 L5,2.5 L0,5 Z" fill={ARROW_ALLY} />
              </marker>
            </defs>
            {hoverArrows.map((arrow, i) => {
              const fromX = hoveredAgentData.position[1] * cellSize + cellSize / 2;
              const fromY = HEADER + hoveredAgentData.position[0] * cellSize + cellSize / 2;
              const toX = arrow.toCol * cellSize + cellSize / 2;
              const toY = HEADER + arrow.toRow * cellSize + cellSize / 2;
              // Shorten arrow so it doesn't overlap agent icons
              const dx = toX - fromX;
              const dy = toY - fromY;
              const len = Math.sqrt(dx * dx + dy * dy);
              const inset = Math.min(cellSize * 0.3, 18);
              const startX = fromX + (dx / len) * inset;
              const startY = fromY + (dy / len) * inset;
              const endX = toX - (dx / len) * inset;
              const endY = toY - (dy / len) * inset;
              const markerId = arrow.type === "attack" ? "arrow-attack" : arrow.type === "ally" ? "arrow-ally" : "arrow-move";
              return (
                <line
                  key={i}
                  x1={startX} y1={startY}
                  x2={endX} y2={endY}
                  stroke={arrow.color}
                  strokeWidth={arrow.type === "attack" ? 2.5 : 1.5}
                  strokeDasharray={arrow.type === "attack" ? "6,4" : "4,4"}
                  markerEnd={`url(#${markerId})`}
                >
                  <animate
                    attributeName="stroke-dashoffset"
                    from="0"
                    to={arrow.type === "attack" ? "-20" : "-16"}
                    dur="0.8s"
                    repeatCount="indefinite"
                  />
                </line>
              );
            })}
          </svg>
        )}

        {/* DOM overlay — agents with CSS transitions for smooth movement */}
        {agentList.map((agent) => {
          if (!agent.alive && !showGhostOutlines) return null;
          const isHovered = hoveredAgent === agent.id;
          const isFaded = hoveredAgent !== null && !isHovered;
          return (
            <div
              key={agent.id}
              className="absolute flex flex-col items-center justify-center"
              style={{
                left: agent.position[1] * cellSize,
                top: HEADER + agent.position[0] * cellSize,
                width: cellSize,
                height: cellSize,
                transition: "left 0.6s ease-in-out, top 0.6s ease-in-out, opacity 0.3s ease, filter 0.3s ease",
                opacity: !agent.alive ? 0.15 : isFaded ? 0.3 : 1,
                filter: isFaded ? "grayscale(0.6)" : "none",
                cursor: agent.alive ? "pointer" : "default",
                zIndex: isHovered ? 10 : 6,
              }}
              onMouseEnter={() => agent.alive && setHoveredAgent(agent.id)}
              onClick={() => agent.alive && setSelectedAgent(agent.id)}
            >
              <div
                style={{
                  width: logoSize,
                  height: logoSize,
                  border: `${TIER_BORDER[agent.tier] || "1px"} solid ${agent.color}`,
                  marginTop: -4,
                  overflow: "hidden",
                  transform: isHovered ? "scale(1.15)" : "scale(1)",
                  transition: "transform 0.2s ease",
                  boxShadow: isHovered ? "0 0 12px rgba(0,0,0,0.15)" : "none",
                }}
              >
                {PROVIDER_LOGOS[agent.provider] ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={PROVIDER_LOGOS[agent.provider]}
                    alt=""
                    style={{ width: "100%", height: "100%", objectFit: "contain" }}
                    draggable={false}
                  />
                ) : (
                  <div style={{ width: "100%", height: "100%", backgroundColor: agent.color }} />
                )}
              </div>
              <span
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  color: !agent.alive ? "rgba(0,0,0,0.15)" : isFaded ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0.7)",
                  textAlign: "center",
                  marginTop: 2,
                  fontFamily: "Inter, system-ui, sans-serif",
                  lineHeight: 1,
                  userSelect: "none",
                  transition: "color 0.3s ease",
                }}
              >
                {agent.name}
              </span>
            </div>
          );
        })}
      </div>
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
