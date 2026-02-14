"use client";

import { useCallback, useRef, useState } from "react";
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
import { useGameState } from "@/hooks/useGameState";
import { getFamilyColor } from "@/lib/colors";
import type { GameInitData, RoundData, AgentState } from "@/lib/types";

export default function ReplayPage() {
  const { rounds, initGame, pushRound } = useGameState();
  const [loaded, setLoaded] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const text = await file.text();
      const data = JSON.parse(text);

      // Parse game.json format
      const config = data.config || {};
      const gameRounds = data.rounds || [];
      const result = data.result || {};

      // Extract agents from first round's data
      const agentsMap: Record<string, Partial<AgentState>> = {};
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

      // Build init
      const initData: GameInitData = {
        type: "game_init",
        grid_size: config.grid_size || 6,
        agents: agentsMap as GameInitData["agents"],
        families: [],
        total_rounds: gameRounds.length,
        result,
      };

      initGame(initData);

      // Push all rounds
      for (const round of gameRounds) {
        const roundData: RoundData = {
          round: round.round,
          grid: { size: config.grid_size || 6, agents: [] },
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
          analysis: {},
          highlights: [],
          alive_count: Object.keys(round.thoughts || {}).length,
          game_over: false,
          winner: null,
        };
        pushRound(roundData);
      }

      // Set to round 1 for stepping
      useGameState.getState().setCurrentRound(1);
      setLoaded(true);
    },
    [initGame, pushRound]
  );

  if (!loaded) {
    return (
      <div className="h-screen flex items-center justify-center bg-white">
        <Card className="w-96">
          <CardHeader>
            <CardTitle>Load Game Replay</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Upload a game.json transcript file to replay the game in the
              dashboard.
            </p>
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
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      <RoundControls />
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
            <div className="h-full p-2">
              <RelationshipWeb />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
