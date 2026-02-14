"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RoundControls } from "@/components/RoundControls";
import { ThoughtStream } from "@/components/ThoughtStream";
import { GameGrid } from "@/components/GameGrid";
import { DeceptionChart } from "@/components/DeceptionChart";
import { KillTimeline } from "@/components/KillTimeline";
import { RelationshipWeb } from "@/components/RelationshipWeb";
import { AgentDetail } from "@/components/AgentDetail";
import { ExportPanel } from "@/components/ExportPanel";
import { useGameState } from "@/hooks/useGameState";
import { FamilyModelPanel } from "@/components/FamilyModelPanel";
import { getReplay } from "@/lib/api";
import type { GameInitData, RoundData, AgentState, FamilyConfig } from "@/lib/types";

function ReplayPageInner() {
  const searchParams = useSearchParams();
  const gameId = searchParams.get("gameId");
  const { initGame, pushRound, selectedAgent } = useGameState();
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadReplayPayload = useCallback(
    async (data: any) => {
      // Store raw JSON for export
      useGameState.getState().setGameJson(data);

      const config = data.config || {};
      const gameRounds = data.rounds || [];
      const result = data.result || {};
      const firstRoundGridAgents = gameRounds[0]?.grid?.agents || [];

      // Extract agents from round data
      const agentsMap: Record<string, Partial<AgentState>> = {};
      for (const agent of firstRoundGridAgents) {
        agentsMap[agent.id] = { ...agent };
      }
      for (const round of gameRounds) {
        for (const msg of round.messages || []) {
          if (msg.sender && !agentsMap[msg.sender]) {
            agentsMap[msg.sender] = {
              id: msg.sender,
              name: msg.sender_name || msg.sender,
              family: msg.family || "",
              provider: "",
              tier: 1,
              alive: true,
            };
          }
        }
        for (const [agentId] of Object.entries(round.thoughts || {})) {
          if (!agentsMap[agentId]) {
            agentsMap[agentId] = {
              id: agentId,
              name: agentId.charAt(0).toUpperCase() + agentId.slice(1),
              family: "",
              tier: 1,
              alive: true,
            };
          }
        }
      }

      // Enrich from family discussions
      for (const round of gameRounds) {
        for (const disc of round.family_discussions || []) {
          for (const entry of disc.transcript || []) {
            const aid = entry.agent_id;
            if (aid && agentsMap[aid]) {
              agentsMap[aid].name = entry.agent;
              agentsMap[aid].tier = entry.tier || 1;
              agentsMap[aid].family = disc.family || agentsMap[aid].family;
            }
          }
        }
      }

      // Enrich from config if present
      const configuredFamilies = (config.families || []) as Array<{
        name: string;
        provider: string;
        color: string;
        agents: Array<{ name: string; tier: number; model: string; temperature: number }>;
      }>;
      for (const family of configuredFamilies) {
        for (const configAgent of family.agents || []) {
          const id = configAgent.name.toLowerCase();
          const existing = agentsMap[id] || {};
          agentsMap[id] = {
            id,
            name: configAgent.name,
            family: family.name,
            provider: family.provider,
            model: configAgent.model,
            tier: configAgent.tier,
            temperature: configAgent.temperature,
            alive: existing.alive ?? true,
            position: existing.position || [0, 0],
            eliminated_by: existing.eliminated_by || null,
            eliminated_round: existing.eliminated_round || null,
          };
        }
      }

      const families: FamilyConfig[] =
        configuredFamilies.length > 0
          ? configuredFamilies.map((family) => ({
              name: family.name,
              provider: family.provider,
              color: family.color,
              agent_ids: (family.agents || []).map((a) => a.name.toLowerCase()),
            }))
          : Object.values(agentsMap).reduce<FamilyConfig[]>((acc, agent) => {
              const family = agent.family || "Unknown";
              const found = acc.find((f) => f.name === family);
              if (found) {
                if (agent.id) found.agent_ids.push(agent.id);
                return acc;
              }
              acc.push({
                name: family,
                provider: agent.provider || "",
                color: "#9CA3AF",
                agent_ids: agent.id ? [agent.id] : [],
              });
              return acc;
            }, []);

      const initData: GameInitData = {
        type: "game_init",
        game_id: data.game_id || gameId || undefined,
        grid_size: config.grid_size || 6,
        agents: agentsMap as GameInitData["agents"],
        families,
        total_rounds: gameRounds.length,
        result,
      };

      initGame(initData);

      // Push all rounds with analysis data if available
      for (const round of gameRounds) {
        const roundData: RoundData = {
          game_id: data.game_id || gameId || undefined,
          round: round.round,
          grid: {
            size: config.grid_size || 6,
            agents: round.grid?.agents || [],
          },
          events: round.events || [],
          thoughts: round.thoughts || {},
          messages: {
            family_discussions: round.family_discussions || [],
            direct_messages: (round.messages || []).filter(
              (m: { channel: string }) => m.channel === "dm"
            ),
            broadcasts: (round.messages || []).filter(
              (m: { channel: string }) => m.channel === "broadcast"
            ),
            family_messages: (round.messages || []).filter(
              (m: { channel: string }) => m.channel === "family"
            ),
          },
          analysis: round.analysis || {},
          highlights: round.highlights || [],
          alive_count:
            round.grid?.agents?.filter((a: { alive: boolean }) => a.alive).length ??
            Object.keys(round.thoughts || {}).length,
          game_over: false,
          winner: null,
        };
        pushRound(roundData);
      }

      useGameState.getState().setCurrentRound(1);
      if (result.winner_name) {
        useGameState.getState().setGameOver(result.winner_name, result.final_reflection);
      }
      setLoaded(true);
    },
    [gameId, initGame, pushRound],
  );

  const handleFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const text = await file.text();
      const data = JSON.parse(text);
      await loadReplayPayload(data);
    },
    [loadReplayPayload]
  );

  useEffect(() => {
    if (!gameId) return;
    const loadById = async () => {
      try {
        setLoadError(null);
        const payload = await getReplay(gameId);
        await loadReplayPayload(payload);
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : "Failed to load replay.");
      }
    };
    void loadById();
  }, [gameId, loadReplayPayload]);

  if (!loaded) {
    return (
      <div className="h-screen flex items-center justify-center bg-white">
        <Card className="w-[420px]">
          <CardHeader>
            <CardTitle className="text-lg">Load Game Replay</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Upload a <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">game.json</code> transcript
              file to replay the game in the dashboard. Step through rounds,
              inspect agent thoughts, and export analysis.
            </p>
            {loadError && (
              <p className="text-xs text-red-600">{loadError}</p>
            )}
            <input
              ref={fileRef}
              type="file"
              accept=".json"
              onChange={handleFile}
              className="hidden"
            />
            <Button onClick={() => fileRef.current?.click()} className="w-full">
              Select game.json
            </Button>
            <p className="text-xs text-muted-foreground text-center">
              Game transcripts are saved in <code className="bg-gray-100 px-1 py-0.5 rounded">data/games/</code>
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      <RoundControls status="replay" />
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          <ResizablePanel defaultSize={30} minSize={20}>
            <div className="h-full p-2">
              <ThoughtStream />
            </div>
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={40} minSize={25}>
            <div className="h-full flex flex-col p-2 gap-2">
              <div className="flex-1 min-h-0">
                <GameGrid />
              </div>
              <div className="h-[220px] shrink-0">
                <DeceptionChart />
              </div>
              <div className="h-[60px] shrink-0">
                <KillTimeline />
              </div>
            </div>
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={30} minSize={15}>
            <div className="h-full p-2 flex flex-col gap-2">
              <div className="h-[260px] shrink-0">
                <FamilyModelPanel />
              </div>
              <div className="flex-1 min-h-0">
                {selectedAgent ? <AgentDetail /> : <RelationshipWeb />}
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
      <div className="border-t px-4 py-2 flex items-center justify-end bg-white">
        <ExportPanel />
      </div>
    </div>
  );
}

export default function ReplayPage() {
  return (
    <Suspense fallback={<div className="h-screen flex items-center justify-center bg-white text-sm text-muted-foreground">Loading replay...</div>}>
      <ReplayPageInner />
    </Suspense>
  );
}
