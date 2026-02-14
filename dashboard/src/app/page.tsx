"use client";

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { RoundControls } from "@/components/RoundControls";
import { ThoughtStream } from "@/components/ThoughtStream";
import { GameGrid } from "@/components/GameGrid";
import { DeceptionChart } from "@/components/DeceptionChart";
import { KillTimeline } from "@/components/KillTimeline";
import { RelationshipWeb } from "@/components/RelationshipWeb";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useGameState } from "@/hooks/useGameState";

export default function DashboardPage() {
  useWebSocket();
  const { rounds } = useGameState();

  return (
    <div className="h-screen flex flex-col bg-white">
      <RoundControls />
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          {/* Left: Thought Stream */}
          <ResizablePanel defaultSize={30} minSize={20}>
            <div className="h-full p-2">
              <ThoughtStream />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Center: Grid + Charts */}
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

          {/* Right: Relationships */}
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
