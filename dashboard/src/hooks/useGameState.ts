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
  playbackSpeed: 1 | 2 | 5 | 10;
  selectedAgent: string | null; // agent id for detail view
  selectedFamily: string | null;
  focusedAgentIds: string[];
  channelFilter: "all" | "thoughts" | "family" | "dm" | "broadcast";
  highlightsOnly: boolean;
  searchQuery: string;
  viewMode: "board" | "relationships";
  showAdjacencyLines: boolean;
  showGhostOutlines: boolean;
  gameJson: unknown | null; // raw game data for export
  activeGameId: string | null;

  // Streaming state
  streamingPhase: string | null;
  streamingRound: number;
  streamingTokens: Record<string, string>; // agent_id -> accumulated text

  // Actions
  initGame: (data: GameInitData) => void;
  pushRound: (data: RoundData) => void;
  setCurrentRound: (n: number) => void;
  stepForward: () => void;
  stepBack: () => void;
  setPlaying: (p: boolean) => void;
  setSpeed: (s: number) => void;
  setPlaybackSpeed: (s: 1 | 2 | 5 | 10) => void;
  setGameOver: (winner: string | null, reflection: string | null) => void;
  setSelectedAgent: (id: string | null) => void;
  setSelectedFamily: (family: string | null) => void;
  setFocusedAgentIds: (ids: string[]) => void;
  toggleFocusedAgentId: (id: string) => void;
  setChannelFilter: (filter: "all" | "thoughts" | "family" | "dm" | "broadcast") => void;
  setHighlightsOnly: (on: boolean) => void;
  setSearchQuery: (q: string) => void;
  setViewMode: (mode: "board" | "relationships") => void;
  setShowAdjacencyLines: (on: boolean) => void;
  setShowGhostOutlines: (on: boolean) => void;
  setGameJson: (data: unknown) => void;
  setActiveGameId: (id: string | null) => void;
  setStreamingPhase: (phase: string | null, round?: number) => void;
  appendToken: (agentId: string, delta: string) => void;
  clearStreaming: () => void;
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
  playbackSpeed: 1,
  selectedAgent: null,
  selectedFamily: null,
  focusedAgentIds: [],
  channelFilter: "all",
  highlightsOnly: false,
  searchQuery: "",
  viewMode: "board",
  showAdjacencyLines: true,
  showGhostOutlines: true,
  streamingPhase: null,
  streamingRound: 0,
  streamingTokens: {},
  gameJson: null,
  activeGameId: null,

  initGame: (data) => {
    const agents: Record<string, AgentState> = {};
    for (const [id, a] of Object.entries(data.agents)) {
      agents[id] = {
        ...a,
        color: getFamilyColor(a.family),
        model: a.model || "",
        temperature: a.temperature ?? 0.7,
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
      activeGameId: data.game_id || null,
      focusedAgentIds: [],
      channelFilter: "all",
      highlightsOnly: false,
      searchQuery: "",
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
  setPlaybackSpeed: (s) => set({ playbackSpeed: s, speed: Math.max(100, Math.floor(1000 / s)) }),
  setGameOver: (winner, reflection) =>
    set({ gameOver: true, winner, finalReflection: reflection }),
  setSelectedAgent: (id) => set({ selectedAgent: id }),
  setSelectedFamily: (family) => set({ selectedFamily: family }),
  setFocusedAgentIds: (ids) => set({ focusedAgentIds: ids }),
  toggleFocusedAgentId: (id) =>
    set((s) => ({
      focusedAgentIds: s.focusedAgentIds.includes(id)
        ? s.focusedAgentIds.filter((x) => x !== id)
        : [...s.focusedAgentIds, id],
    })),
  setChannelFilter: (filter) => set({ channelFilter: filter }),
  setHighlightsOnly: (on) => set({ highlightsOnly: on }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setShowAdjacencyLines: (on) => set({ showAdjacencyLines: on }),
  setShowGhostOutlines: (on) => set({ showGhostOutlines: on }),
  setGameJson: (data) => set({ gameJson: data }),
  setActiveGameId: (id) => set({ activeGameId: id }),
  setStreamingPhase: (phase, round) => set({ streamingPhase: phase, streamingRound: round ?? get().streamingRound }),
  appendToken: (agentId, delta) => set((s) => ({
    streamingTokens: {
      ...s.streamingTokens,
      [agentId]: (s.streamingTokens[agentId] || "") + delta,
    },
  })),
  clearStreaming: () => set({ streamingPhase: null, streamingTokens: {}, streamingRound: 0 }),
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
      playbackSpeed: 1,
      selectedAgent: null,
      selectedFamily: null,
      focusedAgentIds: [],
      channelFilter: "all",
      highlightsOnly: false,
      searchQuery: "",
      viewMode: "board",
      showAdjacencyLines: true,
      showGhostOutlines: true,
      gameJson: null,
      activeGameId: null,
      streamingPhase: null,
      streamingRound: 0,
      streamingTokens: {},
    }),
}));
