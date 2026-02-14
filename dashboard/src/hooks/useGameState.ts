"use client";

import { create } from "zustand";
import type {
  AgentState,
  FamilyConfig,
  GameInitData,
  RoundData,
} from "@/lib/types";
import { getFamilyColor } from "@/lib/colors";

interface GameStore {
  // State
  rounds: RoundData[];
  currentRound: number;
  agents: Record<string, AgentState>;
  families: FamilyConfig[];
  gridSize: number;
  gameOver: boolean;
  winner: string | null;
  finalReflection: string | null;
  totalRounds: number | null;
  playing: boolean;
  speed: number; // ms between rounds in auto-play

  // Actions
  initGame: (data: GameInitData) => void;
  pushRound: (data: RoundData) => void;
  setCurrentRound: (n: number) => void;
  stepForward: () => void;
  stepBack: () => void;
  setPlaying: (p: boolean) => void;
  setSpeed: (s: number) => void;
  setGameOver: (winner: string | null, reflection: string | null) => void;
  reset: () => void;
}

export const useGameState = create<GameStore>((set, get) => ({
  rounds: [],
  currentRound: 0,
  agents: {},
  families: [],
  gridSize: 6,
  gameOver: false,
  winner: null,
  finalReflection: null,
  totalRounds: null,
  playing: false,
  speed: 2000,

  initGame: (data) => {
    const agents: Record<string, AgentState> = {};
    for (const [id, a] of Object.entries(data.agents)) {
      agents[id] = {
        ...a,
        color: getFamilyColor(a.family),
        eliminated_by: null,
        eliminated_round: null,
      } as AgentState;
    }
    set({
      agents,
      families: data.families,
      gridSize: data.grid_size,
      rounds: [],
      currentRound: 0,
      gameOver: false,
      winner: null,
      finalReflection: null,
      totalRounds: data.total_rounds || null,
    });
  },

  pushRound: (data) => {
    // Update agent states from grid data
    const updatedAgents = { ...get().agents };
    if (data.grid?.agents) {
      for (const a of data.grid.agents) {
        if (updatedAgents[a.id]) {
          updatedAgents[a.id] = { ...updatedAgents[a.id], ...a };
        }
      }
    }

    set((s) => ({
      rounds: [...s.rounds, data],
      currentRound: s.rounds.length + 1,
      agents: updatedAgents,
      gameOver: data.game_over,
      winner: data.winner,
    }));
  },

  setCurrentRound: (n) => set({ currentRound: Math.max(0, n) }),
  stepForward: () =>
    set((s) => ({
      currentRound: Math.min(s.currentRound + 1, s.rounds.length),
    })),
  stepBack: () =>
    set((s) => ({ currentRound: Math.max(0, s.currentRound - 1) })),
  setPlaying: (p) => set({ playing: p }),
  setSpeed: (s) => set({ speed: s }),
  setGameOver: (winner, reflection) =>
    set({ gameOver: true, winner, finalReflection: reflection }),
  reset: () =>
    set({
      rounds: [],
      currentRound: 0,
      agents: {},
      families: [],
      gameOver: false,
      winner: null,
      finalReflection: null,
      totalRounds: null,
      playing: false,
    }),
}));
