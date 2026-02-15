/**
 * Heuristic-based optimal move analysis for MARKOV agents.
 * Pure computation — no LLM calls, runs instantly from board state.
 */

import type { AgentState } from "./types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MoveRecommendation {
  agentId: string;
  action: "move" | "eliminate" | "stay";
  target?: [number, number]; // destination cell (move) or target position (eliminate)
  targetName?: string; // name of elimination target
  direction?: string; // human-readable direction label
  score: number; // 0–1 confidence
  rationale: string; // one-line explanation
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DIRECTIONS: [number, number, string][] = [
  [-1, 0, "north"],
  [1, 0, "south"],
  [0, -1, "west"],
  [0, 1, "east"],
  [-1, -1, "northwest"],
  [-1, 1, "northeast"],
  [1, -1, "southwest"],
  [1, 1, "southeast"],
];

function inBounds(r: number, c: number, size: number): boolean {
  return r >= 0 && r < size && c >= 0 && c < size;
}

function manhattan(a: [number, number], b: [number, number]): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

function chebyshev(a: [number, number], b: [number, number]): number {
  return Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
}

/** Distance from cell to grid center (lower = more central). */
function centerDistance(r: number, c: number, size: number): number {
  const center = (size - 1) / 2;
  return Math.max(Math.abs(r - center), Math.abs(c - center));
}

// ---------------------------------------------------------------------------
// Occupancy map
// ---------------------------------------------------------------------------

type OccMap = Map<string, AgentState>;

function buildOccupancy(agents: AgentState[]): OccMap {
  const map: OccMap = new Map();
  for (const a of agents) {
    if (a.alive) map.set(`${a.position[0]},${a.position[1]}`, a);
  }
  return map;
}

function key(r: number, c: number): string {
  return `${r},${c}`;
}

// ---------------------------------------------------------------------------
// Per-agent context
// ---------------------------------------------------------------------------

interface AgentContext {
  adjacentEnemies: AgentState[];
  adjacentAllies: AgentState[];
  nearbyFamily: AgentState[]; // within chebyshev distance 2-3
  totalEnemiesOnBoard: number;
  totalAlliesOnBoard: number;
}

function getContext(
  agent: AgentState,
  alive: AgentState[],
  occ: OccMap,
  gridSize: number,
): AgentContext {
  const [row, col] = agent.position;
  const adjacentEnemies: AgentState[] = [];
  const adjacentAllies: AgentState[] = [];
  const nearbyFamily: AgentState[] = [];

  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = row + dr;
      const nc = col + dc;
      if (!inBounds(nr, nc, gridSize)) continue;
      const occupant = occ.get(key(nr, nc));
      if (!occupant) continue;
      if (occupant.family === agent.family) {
        adjacentAllies.push(occupant);
      } else {
        adjacentEnemies.push(occupant);
      }
    }
  }

  for (const a of alive) {
    if (a.id === agent.id || a.family !== agent.family) continue;
    const dist = chebyshev(agent.position, a.position);
    if (dist >= 2 && dist <= 3) nearbyFamily.push(a);
  }

  const totalAlliesOnBoard = alive.filter(
    (a) => a.family === agent.family && a.id !== agent.id,
  ).length;
  const totalEnemiesOnBoard = alive.filter(
    (a) => a.family !== agent.family,
  ).length;

  return {
    adjacentEnemies,
    adjacentAllies,
    nearbyFamily,
    totalEnemiesOnBoard,
    totalAlliesOnBoard,
  };
}

// ---------------------------------------------------------------------------
// Score individual actions
// ---------------------------------------------------------------------------

interface ScoredAction {
  action: "move" | "eliminate" | "stay";
  target?: [number, number];
  targetName?: string;
  direction?: string;
  score: number;
  rationale: string;
}

function scoreElimination(
  agent: AgentState,
  enemy: AgentState,
  ctx: AgentContext,
  occ: OccMap,
  gridSize: number,
): ScoredAction {
  let score = 0.8;
  const reasons: string[] = [];

  // Bonus: target is isolated (no allies of their family adjacent)
  let targetAllyCount = 0;
  const [er, ec] = enemy.position;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = er + dr;
      const nc = ec + dc;
      if (!inBounds(nr, nc, gridSize)) continue;
      const occ2 = occ.get(key(nr, nc));
      if (occ2 && occ2.family === enemy.family && occ2.id !== enemy.id) {
        targetAllyCount++;
      }
    }
  }
  if (targetAllyCount === 0) {
    score += 0.1;
    reasons.push("isolated target");
  }

  // Bonus: target is higher tier (more valuable to eliminate)
  if (enemy.tier < agent.tier) {
    score += 0.1;
    reasons.push(`high-value (tier ${enemy.tier})`);
  }

  // Penalty: retaliation risk (other enemies adjacent to us)
  const otherAdjacentEnemies = ctx.adjacentEnemies.filter(
    (e) => e.id !== enemy.id,
  ).length;
  if (otherAdjacentEnemies > 0) {
    score -= 0.3;
    reasons.push(`retaliation risk (${otherAdjacentEnemies} other threats)`);
  }

  const detail = reasons.length > 0 ? ` (${reasons.join(", ")})` : "";
  return {
    action: "eliminate",
    target: enemy.position,
    targetName: enemy.name,
    score: Math.max(0, Math.min(1, score)),
    rationale: `Eliminate ${enemy.name}${detail}`,
  };
}

function scoreMove(
  agent: AgentState,
  dr: number,
  dc: number,
  dirLabel: string,
  ctx: AgentContext,
  occ: OccMap,
  gridSize: number,
  alive: AgentState[],
): ScoredAction | null {
  const nr = agent.position[0] + dr;
  const nc = agent.position[1] + dc;
  if (!inBounds(nr, nc, gridSize)) return null;
  if (occ.has(key(nr, nc))) return null; // cell occupied

  let score = 0.1; // baseline for any valid move
  const reasons: string[] = [];

  // Positional: moving toward center
  const currentCenterDist = centerDistance(
    agent.position[0],
    agent.position[1],
    gridSize,
  );
  const newCenterDist = centerDistance(nr, nc, gridSize);
  if (newCenterDist < currentCenterDist) {
    score += 0.2;
    reasons.push("toward center");
  }

  // Flee: moving away from threats
  if (ctx.adjacentEnemies.length >= 2) {
    // Check if this move increases distance from enemies
    let distIncrease = 0;
    for (const enemy of ctx.adjacentEnemies) {
      const oldDist = chebyshev(agent.position, enemy.position);
      const newDist = chebyshev([nr, nc], enemy.position);
      if (newDist > oldDist) distIncrease++;
    }
    if (distIncrease > 0) {
      score += 0.4;
      reasons.push(`away from ${ctx.adjacentEnemies.length} threats`);
    }
  }

  // Family clustering: moving toward nearby family
  if (ctx.nearbyFamily.length > 0) {
    let closerCount = 0;
    for (const fam of ctx.nearbyFamily) {
      const oldDist = chebyshev(agent.position, fam.position);
      const newDist = chebyshev([nr, nc], fam.position);
      if (newDist < oldDist) closerCount++;
    }
    if (closerCount > 0) {
      score += 0.3;
      reasons.push("toward family");
    }
  }

  // Check: does this move put us adjacent to new enemies?
  let newAdjacentEnemies = 0;
  for (let dr2 = -1; dr2 <= 1; dr2++) {
    for (let dc2 = -1; dc2 <= 1; dc2++) {
      if (dr2 === 0 && dc2 === 0) continue;
      const checkR = nr + dr2;
      const checkC = nc + dc2;
      if (!inBounds(checkR, checkC, gridSize)) continue;
      const occupant = occ.get(key(checkR, checkC));
      if (occupant && occupant.family !== agent.family) {
        newAdjacentEnemies++;
      }
    }
  }
  if (newAdjacentEnemies > ctx.adjacentEnemies.length) {
    score -= 0.25;
    reasons.push("enters danger zone");
  }

  return {
    action: "move",
    target: [nr, nc],
    direction: dirLabel,
    score: Math.max(0, Math.min(1, score)),
    rationale:
      reasons.length > 0
        ? `Move ${dirLabel} (${reasons.join(", ")})`
        : `Move ${dirLabel}`,
  };
}

function scoreStay(
  agent: AgentState,
  ctx: AgentContext,
): ScoredAction {
  let score = 0.3;
  const reasons: string[] = [];

  if (ctx.adjacentEnemies.length === 0) {
    score += 0.1;
    reasons.push("no immediate threats");
  } else {
    score -= 0.2;
    reasons.push(`${ctx.adjacentEnemies.length} threats adjacent`);
  }

  if (ctx.adjacentAllies.length >= 2) {
    score += 0.2;
    reasons.push("protected by family");
  }

  return {
    action: "stay",
    score: Math.max(0, Math.min(1, score)),
    rationale:
      reasons.length > 0
        ? `Stay put (${reasons.join(", ")})`
        : "Stay put",
  };
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export function computeOptimalMoves(
  agents: AgentState[],
  gridSize: number,
): Record<string, MoveRecommendation> {
  const alive = agents.filter((a) => a.alive);
  const occ = buildOccupancy(alive);
  const result: Record<string, MoveRecommendation> = {};

  for (const agent of alive) {
    const ctx = getContext(agent, alive, occ, gridSize);
    const candidates: ScoredAction[] = [];

    // Score eliminations
    for (const enemy of ctx.adjacentEnemies) {
      candidates.push(
        scoreElimination(agent, enemy, ctx, occ, gridSize),
      );
    }

    // Score moves
    for (const [dr, dc, label] of DIRECTIONS) {
      const moveScore = scoreMove(
        agent, dr, dc, label, ctx, occ, gridSize, alive,
      );
      if (moveScore) candidates.push(moveScore);
    }

    // Score stay
    candidates.push(scoreStay(agent, ctx));

    // Pick the highest-scoring action
    candidates.sort((a, b) => b.score - a.score);
    const best = candidates[0];

    result[agent.id] = {
      agentId: agent.id,
      action: best.action,
      target: best.target,
      targetName: best.targetName,
      direction: best.direction,
      score: best.score,
      rationale: best.rationale,
    };
  }

  return result;
}
