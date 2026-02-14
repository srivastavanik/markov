"use client";

import { useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AgentBadge } from "./AgentBadge";
import { useGameState } from "@/hooks/useGameState";
import { getFamilyColor, SEVERITY_COLORS } from "@/lib/colors";
import type { AgentAnalysis, HighlightData, MessageData, RoundData } from "@/lib/types";

export function ThoughtStream() {
  const { rounds, currentRound, agents } = useGameState();
  const endRef = useRef<HTMLDivElement>(null);

  const visibleRounds = rounds.slice(0, currentRound);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentRound]);

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="py-3 px-4 shrink-0">
        <CardTitle className="text-sm font-semibold">Thought Stream</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 p-0">
        <ScrollArea className="h-full px-4 pb-4">
          {visibleRounds.length === 0 && (
            <p className="text-sm text-muted-foreground py-8 text-center">
              Waiting for game data...
            </p>
          )}
          {visibleRounds.map((round) => (
            <RoundSection key={round.round} round={round} agents={agents} />
          ))}
          <div ref={endRef} />
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function RoundSection({
  round,
  agents,
}: {
  round: RoundData;
  agents: Record<string, import("@/lib/types").AgentState>;
}) {
  const thoughts = round.thoughts || {};
  const analysis = round.analysis || {};
  const highlights = round.highlights || [];
  const messages = round.messages || { broadcasts: [], direct_messages: [], family_messages: [], family_discussions: [] };

  return (
    <div className="mb-4">
      <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 mt-3">
        Round {round.round}
      </div>

      {/* Highlights for this round */}
      {highlights.map((h, i) => (
        <HighlightBanner key={i} highlight={h} />
      ))}

      {/* Agent entries: thought + what they said */}
      {Object.entries(thoughts).map(([agentId, thought]) => {
        const agent = agents[agentId];
        if (!agent) return null;
        const agentAnalysis = analysis[agentId];
        const agentMessages = getAllAgentMessages(agentId, messages);

        return (
          <ThoughtEntry
            key={agentId}
            agentId={agentId}
            agentName={agent.name}
            family={agent.family}
            tier={agent.tier}
            color={getFamilyColor(agent.family)}
            alive={agent.alive}
            thought={thought}
            analysis={agentAnalysis}
            messagesSent={agentMessages}
          />
        );
      })}
    </div>
  );
}

function ThoughtEntry({
  agentName,
  family,
  tier,
  color,
  alive,
  thought,
  analysis,
  messagesSent,
}: {
  agentId: string;
  agentName: string;
  family: string;
  tier: number;
  color: string;
  alive: boolean;
  thought: string;
  analysis?: AgentAnalysis;
  messagesSent: MessageData[];
}) {
  return (
    <div className="mb-3">
      {/* Thought */}
      <div
        className="rounded-md p-3 bg-gray-50"
        style={{ borderLeft: `3px solid ${color}` }}
      >
        <div className="flex items-center justify-between mb-1.5">
          <AgentBadge name={agentName} family={family} color={color} tier={tier} alive={alive} small />
          {analysis && <AnalysisBadges analysis={analysis} />}
        </div>
        <p className="text-xs leading-relaxed text-gray-700 font-mono whitespace-pre-wrap">
          {thought}
        </p>
      </div>

      {/* Messages sent */}
      {messagesSent.length > 0 && (
        <div className="ml-4 mt-1.5 space-y-1">
          {messagesSent.map((msg, i) => (
            <div
              key={i}
              className="text-xs text-gray-500 rounded px-2 py-1 bg-gray-50/50"
            >
              {msg.channel === "broadcast" && (
                <span>
                  <span className="font-medium text-gray-600">{agentName}</span>{" "}
                  [broadcast]: &ldquo;{truncate(msg.content, 200)}&rdquo;
                </span>
              )}
              {msg.channel === "dm" && (
                <span>
                  <span className="font-medium text-gray-600">{agentName}</span>{" "}
                  &rarr; {msg.recipient} [DM]: &ldquo;{truncate(msg.content, 200)}&rdquo;
                </span>
              )}
              {msg.channel === "family" && (
                <span>
                  <span className="font-medium text-gray-600">{agentName}</span>{" "}
                  [house]: &ldquo;{truncate(msg.content, 200)}&rdquo;
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AnalysisBadges({ analysis }: { analysis: AgentAnalysis }) {
  const badges: Array<{ label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = [];

  if (analysis.malice?.elimination_planning) {
    badges.push({ label: "Malice", variant: "destructive" });
  }
  if (analysis.betrayal?.detected) {
    badges.push({ label: "Betrayal", variant: "destructive" });
  }
  if (analysis.deception_delta > 0.5) {
    badges.push({ label: `D:${analysis.deception_delta.toFixed(2)}`, variant: "secondary" });
  }
  if (analysis.safety_artifacts?.hedging || analysis.safety_artifacts?.refusal_language) {
    badges.push({ label: "Safety", variant: "outline" });
  }
  if (analysis.betrayal?.guilt_expressed) {
    badges.push({ label: "Guilt", variant: "outline" });
  }

  if (badges.length === 0) return null;

  return (
    <div className="flex gap-1">
      {badges.map((b, i) => (
        <Badge key={i} variant={b.variant} className="text-[10px] px-1.5 py-0 h-5">
          {b.label}
        </Badge>
      ))}
    </div>
  );
}

function HighlightBanner({ highlight }: { highlight: HighlightData }) {
  const borderColor = SEVERITY_COLORS[highlight.severity] || "#9CA3AF";
  return (
    <div
      className="text-xs rounded-md px-3 py-2 mb-2 bg-amber-50"
      style={{ borderLeft: `3px solid ${borderColor}` }}
    >
      <span className="font-semibold" style={{ color: borderColor }}>
        {highlight.severity.toUpperCase()}
      </span>{" "}
      <span className="text-gray-700">{highlight.description}</span>
      {highlight.excerpt && (
        <p className="text-gray-500 mt-0.5 italic">&ldquo;{truncate(highlight.excerpt, 150)}&rdquo;</p>
      )}
    </div>
  );
}

function getAllAgentMessages(agentId: string, messages: RoundData["messages"]): MessageData[] {
  const all: MessageData[] = [];
  for (const m of messages.broadcasts || []) {
    if (m.sender === agentId) all.push(m);
  }
  for (const m of messages.direct_messages || []) {
    if (m.sender === agentId) all.push(m);
  }
  return all;
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + "...";
}
