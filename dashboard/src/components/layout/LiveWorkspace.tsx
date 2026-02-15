"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import Image from "next/image";
import { GameGrid } from "@/components/GameGrid";
import { KillTimeline } from "@/components/KillTimeline";
import { useGameState } from "@/hooks/useGameState";
import type { RoundData } from "@/lib/types";

const PROVIDER_LOGOS: Record<string, string> = {
  anthropic: "/logos/anthropic.png",
  openai: "/logos/openai.webp",
  google: "/logos/google.png",
  xai: "/logos/xai.png",
};

const TIER_LABELS: Record<number, string> = { 1: "Boss", 2: "Lt", 3: "Soldier" };

type MsgTab = "thoughts" | "family" | "dms" | "broadcasts" | "all";

export function LiveWorkspace() {
  const { rounds, agents, families } = useGameState();
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [msgTab, setMsgTab] = useState<MsgTab>("all");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new rounds arrive
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [rounds.length]);

  const agentsByFamily = useMemo(() => {
    const map: Record<string, typeof agents[string][]> = {};
    for (const a of Object.values(agents)) {
      (map[a.family] ??= []).push(a);
    }
    return map;
  }, [agents]);

  const activeFamily = selectedFamily ?? families[0]?.name ?? null;
  const familyAgents = activeFamily ? (agentsByFamily[activeFamily] ?? []) : [];
  const familyConfig = families.find((f) => f.name === activeFamily);
  const provider = familyConfig?.provider ?? "";
  const familyAgentIds = new Set(familyAgents.map((a) => a.id));

  const activeAgent = selectedAgent ? (agents[selectedAgent] ?? null) : null;

  // Build a flat stream of entries across ALL rounds
  const stream = useMemo(() => {
    const entries: StreamEntry[] = [];
    for (const round of rounds) {
      const rn = round.round;

      // Thoughts
      if (round.thoughts) {
        for (const [agentId, thought] of Object.entries(round.thoughts)) {
          const agent = agents[agentId];
          if (!agent) continue;
          // Filter by selected agent or family
          if (activeAgent && agent.id !== activeAgent.id) continue;
          if (!activeAgent && !familyAgentIds.has(agent.id)) continue;

          entries.push({
            round: rn, type: "thought", agentId, agentName: agent.name,
            agentProvider: agent.provider, content: thought,
          });
        }
      }

      // Family discussions
      const discussions = round.messages?.family_discussions ?? [];
      for (const disc of discussions) {
        if (disc.family !== activeFamily) continue;
        for (const entry of disc.transcript ?? []) {
          if (activeAgent && entry.agent_id !== activeAgent.id) continue;
          entries.push({
            round: rn, type: "family", agentId: entry.agent_id,
            agentName: entry.agent, agentProvider: provider,
            content: entry.content, meta: `Tier ${entry.tier}`,
          });
        }
      }

      // DMs
      const dms = round.messages?.direct_messages ?? [];
      for (const dm of dms) {
        const isRelevant = activeAgent
          ? dm.sender === activeAgent.id || (dm.recipient && dm.recipient.toLowerCase() === activeAgent.name.toLowerCase())
          : familyAgentIds.has(dm.sender) || familyAgents.some((a) => dm.recipient && dm.recipient.toLowerCase() === a.name.toLowerCase());
        if (!isRelevant) continue;
        const senderAgent = Object.values(agents).find((a) => a.id === dm.sender);
        entries.push({
          round: rn, type: "dm", agentId: dm.sender,
          agentName: dm.sender_name, agentProvider: senderAgent?.provider ?? "",
          content: dm.content, meta: `-> ${dm.recipient}`,
        });
      }

      // Broadcasts
      const bcast = round.messages?.broadcasts ?? [];
      for (const msg of bcast) {
        if (activeAgent && msg.sender !== activeAgent.id) continue;
        // Show all broadcasts if no specific agent selected
        const senderAgent = Object.values(agents).find((a) => a.id === msg.sender);
        entries.push({
          round: rn, type: "broadcast", agentId: msg.sender,
          agentName: msg.sender_name, agentProvider: senderAgent?.provider ?? "",
          content: msg.content,
        });
      }
    }
    return entries;
  }, [rounds, agents, activeAgent, activeFamily, familyAgentIds, familyAgents, provider]);

  // Filter by tab
  const filtered = msgTab === "all" ? stream : stream.filter((e) => e.type === msgTab);

  // Group by round for display
  const byRound = useMemo(() => {
    const map = new Map<number, StreamEntry[]>();
    for (const e of filtered) {
      const arr = map.get(e.round) ?? [];
      arr.push(e);
      map.set(e.round, arr);
    }
    return Array.from(map.entries()).sort((a, b) => a[0] - b[0]);
  }, [filtered]);

  return (
    <div className="flex-1 min-h-0 flex">
      {/* LEFT: Grid */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-1 min-h-0">
          <GameGrid />
        </div>
        <div className="h-[80px] shrink-0 border-t border-black/5">
          <KillTimeline />
        </div>
      </div>

      {/* RIGHT: Stream panel */}
      <div className="w-[440px] shrink-0 flex flex-col h-full overflow-hidden border-l border-black/10">
        {/* Family tabs */}
        <div className="flex shrink-0">
          {families.map((fam) => (
            <button
              key={fam.name}
              onClick={() => { setSelectedFamily(fam.name); setSelectedAgent(null); }}
              className={`flex-1 flex items-center justify-center gap-1.5 px-1 py-2 text-[11px] font-medium border-b-2 transition-colors ${
                activeFamily === fam.name ? "border-black text-black" : "border-transparent text-black/35 hover:text-black/60"
              }`}
            >
              {PROVIDER_LOGOS[fam.provider] && (
                <Image src={PROVIDER_LOGOS[fam.provider]} alt="" width={14} height={14} className="object-contain" />
              )}
              {fam.name}
            </button>
          ))}
        </div>

        {/* Agent tabs within family */}
        <div className="flex shrink-0 bg-black/[0.02] border-b border-black/10">
          <button
            onClick={() => setSelectedAgent(null)}
            className={`px-3 py-1 text-[10px] border-b-2 ${!activeAgent ? "border-black text-black font-medium" : "border-transparent text-black/40"}`}
          >
            All
          </button>
          {familyAgents.sort((a, b) => a.tier - b.tier).map((agent) => (
            <button
              key={agent.id}
              onClick={() => setSelectedAgent(agent.id)}
              className={`flex-1 px-1 py-1 text-[10px] border-b-2 transition-colors ${
                activeAgent?.id === agent.id ? "border-black text-black font-medium" : "border-transparent text-black/40"
              }`}
            >
              {agent.name}
              <span className="text-black/25 ml-0.5">{TIER_LABELS[agent.tier]}</span>
              {!agent.alive && <span className="text-red-400 ml-0.5">X</span>}
            </button>
          ))}
        </div>

        {/* Message type tabs */}
        <div className="flex shrink-0 border-b border-black/10">
          {(["all", "thoughts", "family", "dms", "broadcasts"] as MsgTab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setMsgTab(tab)}
              className={`flex-1 px-1 py-1 text-[10px] capitalize border-b-2 transition-colors ${
                msgTab === tab ? "border-black text-black font-medium" : "border-transparent text-black/35"
              }`}
            >
              {tab === "dms" ? "DMs" : tab === "all" ? "All" : tab}
            </button>
          ))}
        </div>

        {/* Scrollable stream */}
        <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto">
          {rounds.length === 0 && (
            <div className="text-xs text-black/30 text-center py-12">
              Waiting for game data...
            </div>
          )}

          {byRound.map(([roundNum, entries]) => (
            <div key={roundNum}>
              {/* Round header */}
              <div className="sticky top-0 z-10 bg-black/[0.03] px-3 py-1 text-[10px] font-medium text-black/50 border-b border-black/5">
                Round {roundNum}
              </div>
              {entries.map((entry, i) => (
                <EntryCard key={`${roundNum}-${entry.type}-${entry.agentId}-${i}`} entry={entry} />
              ))}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="shrink-0 border-t border-black/10 px-3 py-1 text-[10px] text-black/30 flex justify-between">
          <span>{rounds.length} rounds</span>
          <span>{Object.values(agents).filter((a) => a.alive).length}/{Object.values(agents).length} alive</span>
        </div>
      </div>
    </div>
  );
}

interface StreamEntry {
  round: number;
  type: "thought" | "family" | "dm" | "broadcast";
  agentId: string;
  agentName: string;
  agentProvider: string;
  content: string;
  meta?: string;
}

const TYPE_STYLE: Record<string, { label: string; border: string; bg: string }> = {
  thought: { label: "thinking", border: "border-violet-200", bg: "bg-violet-50/40" },
  family: { label: "family", border: "border-blue-200", bg: "bg-blue-50/30" },
  dm: { label: "DM", border: "border-amber-200", bg: "bg-amber-50/30" },
  broadcast: { label: "broadcast", border: "border-black/10", bg: "" },
};

function EntryCard({ entry }: { entry: StreamEntry }) {
  const style = TYPE_STYLE[entry.type] ?? TYPE_STYLE.broadcast;
  return (
    <div className={`border-b border-black/5 px-3 py-2 ${style.bg}`}>
      <div className="flex items-center gap-1.5 mb-1">
        {PROVIDER_LOGOS[entry.agentProvider] && (
          <Image src={PROVIDER_LOGOS[entry.agentProvider]} alt="" width={12} height={12} className="object-contain" />
        )}
        <span className="text-[11px] font-medium text-black/80">{entry.agentName}</span>
        <span className={`text-[9px] px-1 py-px ${style.border} border text-black/40`}>
          {style.label}
        </span>
        {entry.meta && <span className="text-[9px] text-black/30">{entry.meta}</span>}
      </div>
      <div className="text-[11px] text-black/70 whitespace-pre-wrap leading-relaxed pl-5">
        {entry.content}
      </div>
    </div>
  );
}
