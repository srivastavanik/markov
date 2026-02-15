"use client";

import { useState, useMemo, useRef, useEffect, type ReactNode } from "react";
import Image from "next/image";
import { GameGrid } from "@/components/GameGrid";
import { KillTimeline } from "@/components/KillTimeline";
import { useGameState } from "@/hooks/useGameState";
import type { AgentState } from "@/lib/types";

const PROVIDER_LOGOS: Record<string, string> = {
  anthropic: "/logos/anthropic.png",
  openai: "/logos/openai.webp",
  google: "/logos/google.png",
  xai: "/logos/xai.png",
};

const TIER_LABELS: Record<number, string> = { 1: "Boss", 2: "Lt", 3: "Soldier" };

type MsgTab = "reasoning" | "thoughts" | "family" | "dms" | "broadcasts" | "all";

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

function cleanText(raw: string): string {
  let s = raw;
  s = s.replace(/^#{1,6}\s+/gm, "");
  s = s.replace(/\*\*([^*]+)\*\*/g, "$1");
  s = s.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "$1");
  s = s.replace(/^---+$/gm, "");
  s = s.replace(/^[-*]\s+/gm, "");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

function renderMentions(text: string, agents: Record<string, AgentState>): ReactNode[] {
  const parts = text.split(/(@[\w.+-]+)/g);
  return parts.map((part, i) => {
    if (part.startsWith("@")) {
      const name = part.slice(1);
      const agent = Object.values(agents).find(
        (a) => a.name.toLowerCase() === name.toLowerCase(),
      );
      if (agent) {
        return (
          <span key={i} className="bg-blue-100 text-blue-700 px-1 py-px text-[10px] font-medium">
            @{agent.name}
          </span>
        );
      }
    }
    return <span key={i}>{part}</span>;
  });
}

// ---------------------------------------------------------------------------
// Animation components
// ---------------------------------------------------------------------------

function FadeInText({ text, isNew, agents }: { text: string; isNew: boolean; agents: Record<string, AgentState> }) {
  const cleaned = cleanText(text);
  if (!isNew) {
    return <span>{renderMentions(cleaned, agents)}</span>;
  }
  const words = cleaned.split(/(\s+)/);
  return (
    <span>
      {words.map((word, i) => (
        <span
          key={i}
          className="inline opacity-0 animate-fadeInWord"
          style={{ animationDelay: `${Math.floor(i / 2) * 25}ms` }}
        >
          {word}
        </span>
      ))}
    </span>
  );
}

function StreamingText({ text, agents }: { text: string; agents: Record<string, AgentState> }) {
  const cleaned = cleanText(text);
  return (
    <span>
      {renderMentions(cleaned, agents)}
      <span className="inline-block w-1.5 h-3 bg-black/40 ml-0.5 align-middle animate-blink" />
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function LiveWorkspace() {
  const {
    rounds, agents, families,
    streamingPhase, streamingTokens, streamingRound,
  } = useGameState();
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [msgTab, setMsgTab] = useState<MsgTab>("all");
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastRoundCount = useRef(0);

  // Auto-scroll on new rounds or streaming tokens
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [rounds.length, streamingTokens]);

  // Track which round is "new" for fade animation
  const newRoundNum = rounds.length > lastRoundCount.current ? rounds[rounds.length - 1]?.round : -1;
  useEffect(() => { lastRoundCount.current = rounds.length; }, [rounds.length]);

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

  // Build flat stream across ALL rounds
  const stream = useMemo(() => {
    const entries: StreamEntry[] = [];
    for (const round of rounds) {
      const rn = round.round;

      // Reasoning traces (extended thinking)
      if (round.reasoning_traces) {
        for (const [agentId, trace] of Object.entries(round.reasoning_traces)) {
          const agent = agents[agentId];
          if (!agent) continue;
          if (activeAgent && agent.id !== activeAgent.id) continue;
          if (!activeAgent && !familyAgentIds.has(agent.id)) continue;
          const text = trace.thinking_trace || trace.reasoning_summary || "";
          if (!text) continue;
          const cls = trace.classification;
          entries.push({
            round: rn, type: "reasoning", agentId, agentName: agent.name,
            agentProvider: agent.provider, content: text,
            classification: cls ? {
              intent_tags: cls.intent_tags,
              moral_friction: cls.moral_friction,
              deception_sophistication: cls.deception_sophistication,
              strategic_depth: cls.strategic_depth,
              theory_of_mind: cls.theory_of_mind,
              meta_awareness: cls.meta_awareness,
            } : undefined,
            thinkingTokens: trace.tokens_thinking,
          });
        }
      }

      // Legacy thoughts (fallback if no reasoning traces)
      if (round.thoughts && !round.reasoning_traces) {
        for (const [agentId, thought] of Object.entries(round.thoughts)) {
          const agent = agents[agentId];
          if (!agent) continue;
          if (activeAgent && agent.id !== activeAgent.id) continue;
          if (!activeAgent && !familyAgentIds.has(agent.id)) continue;
          entries.push({ round: rn, type: "thought", agentId, agentName: agent.name, agentProvider: agent.provider, content: thought });
        }
      }

      const discussions = round.messages?.family_discussions ?? [];
      for (const disc of discussions) {
        if (disc.family !== activeFamily) continue;
        for (const entry of disc.transcript ?? []) {
          if (activeAgent && entry.agent_id !== activeAgent.id) continue;
          entries.push({ round: rn, type: "family", agentId: entry.agent_id, agentName: entry.agent, agentProvider: provider, content: entry.content, meta: TIER_LABELS[entry.tier] });
        }
      }

      const dms = round.messages?.direct_messages ?? [];
      for (const dm of dms) {
        // Show DMs involving any agent in selected family (both sent and received)
        const isRelevant = activeAgent
          ? dm.sender === activeAgent.id || (dm.recipient?.toLowerCase() === activeAgent.name.toLowerCase())
          : familyAgentIds.has(dm.sender) || familyAgents.some((a) => dm.recipient?.toLowerCase() === a.name.toLowerCase());
        if (!isRelevant) continue;
        const senderAgent = Object.values(agents).find((a) => a.id === dm.sender);
        const isSent = activeAgent ? dm.sender === activeAgent.id : familyAgentIds.has(dm.sender);
        entries.push({ round: rn, type: "dm", agentId: dm.sender, agentName: dm.sender_name, agentProvider: senderAgent?.provider ?? "", content: dm.content, meta: dm.recipient ?? "", isSent });
      }

      const bcast = round.messages?.broadcasts ?? [];
      for (const msg of bcast) {
        if (activeAgent && msg.sender !== activeAgent.id) continue;
        const senderAgent = Object.values(agents).find((a) => a.id === msg.sender);
        entries.push({ round: rn, type: "broadcast", agentId: msg.sender, agentName: msg.sender_name, agentProvider: senderAgent?.provider ?? "", content: msg.content });
      }
    }
    return entries;
  }, [rounds, agents, activeAgent, activeFamily, familyAgentIds, familyAgents, provider]);

  const tabToType: Record<MsgTab, string> = { all: "all", reasoning: "reasoning", thoughts: "thought", family: "family", dms: "dm", broadcasts: "broadcast" };
  const filtered = msgTab === "all" ? stream : stream.filter((e) => e.type === tabToType[msgTab]);

  const byRound = useMemo(() => {
    const map = new Map<number, StreamEntry[]>();
    for (const e of filtered) {
      (map.get(e.round) ?? (() => { const a: StreamEntry[] = []; map.set(e.round, a); return a; })()).push(e);
    }
    return Array.from(map.entries()).sort((a, b) => a[0] - b[0]);
  }, [filtered]);

  // Build streaming entries for display
  const streamingEntries = useMemo(() => {
    if (!streamingPhase) return [];
    const entries: { agentId: string; agentName: string; agentProvider: string; text: string; phase: string }[] = [];
    for (const [agentId, text] of Object.entries(streamingTokens)) {
      if (!text) continue;
      const agent = agents[agentId];
      if (!agent) continue;
      if (activeAgent && agent.id !== activeAgent.id) continue;
      if (!activeAgent && !familyAgentIds.has(agent.id)) continue;
      entries.push({ agentId, agentName: agent.name, agentProvider: agent.provider, text, phase: streamingPhase });
    }
    return entries;
  }, [streamingPhase, streamingTokens, agents, activeAgent, familyAgentIds]);

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
            <button key={fam.name} onClick={() => { setSelectedFamily(fam.name); setSelectedAgent(null); }}
              className={`flex-1 flex items-center justify-center gap-1.5 px-1 py-2 text-[11px] font-medium border-b-2 transition-colors ${activeFamily === fam.name ? "border-black text-black" : "border-transparent text-black/35 hover:text-black/60"}`}>
              {PROVIDER_LOGOS[fam.provider] && <Image src={PROVIDER_LOGOS[fam.provider]} alt="" width={14} height={14} className="object-contain" />}
              {fam.name}
            </button>
          ))}
        </div>

        {/* Agent tabs */}
        <div className="flex shrink-0 bg-black/[0.02] border-b border-black/10">
          <button onClick={() => setSelectedAgent(null)}
            className={`px-3 py-1 text-[10px] border-b-2 ${!activeAgent ? "border-black text-black font-medium" : "border-transparent text-black/40"}`}>All</button>
          {familyAgents.sort((a, b) => a.tier - b.tier).map((agent) => (
            <button key={agent.id} onClick={() => setSelectedAgent(agent.id)}
              className={`flex-1 px-1 py-1 text-[10px] border-b-2 transition-colors ${activeAgent?.id === agent.id ? "border-black text-black font-medium" : "border-transparent text-black/40"}`}>
              {agent.name}
              <span className="text-black/25 ml-0.5">{TIER_LABELS[agent.tier]}</span>
              {!agent.alive && <span className="text-red-400 ml-0.5">X</span>}
            </button>
          ))}
        </div>

        {/* Message type tabs */}
        <div className="flex shrink-0 border-b border-black/10">
          {(["all", "reasoning", "thoughts", "family", "dms", "broadcasts"] as MsgTab[]).map((tab) => (
            <button key={tab} onClick={() => setMsgTab(tab)}
              className={`flex-1 px-1 py-1.5 text-[10px] capitalize border-b-2 transition-colors ${msgTab === tab ? "border-black text-black font-medium" : "border-transparent text-black/35"}`}>
              {tab === "dms" ? "DMs" : tab === "all" ? "All" : tab}
            </button>
          ))}
        </div>

        {/* Scrollable stream */}
        <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto">
          {rounds.length === 0 && streamingEntries.length === 0 && (
            <div className="text-xs text-black/30 text-center py-12">Waiting for game data...</div>
          )}

          {byRound.map(([roundNum, entries]) => (
            <div key={roundNum}>
              <div className="sticky top-0 z-10 bg-black/[0.03] px-4 py-1.5 text-[10px] font-medium text-black/50 border-b border-black/5">
                Round {roundNum}
              </div>
              <div className="divide-y divide-black/[0.04]">
                {entries.map((entry, i) => (
                  msgTab === "dms"
                    ? <DmBubble key={`${roundNum}-dm-${i}`} entry={entry} activeAgent={activeAgent} agents={agents} isNew={roundNum === newRoundNum} />
                    : <EntryCard key={`${roundNum}-${entry.type}-${entry.agentId}-${i}`} entry={entry} agents={agents} isNew={roundNum === newRoundNum} />
                ))}
              </div>
            </div>
          ))}

          {/* Live streaming section */}
          {streamingEntries.length > 0 && (
            <div>
              <div className="sticky top-0 z-10 bg-violet-100/50 px-4 py-1.5 text-[10px] font-medium text-violet-600 border-b border-violet-200/30 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-violet-500 animate-pulse" />
                {streamingPhase === "deciding" ? "Reasoning..." : streamingPhase === "thinking" ? "Thinking..." : streamingPhase === "family_discussion" ? "Family Discussion..." : "Communicating..."}
              </div>
              <div className="divide-y divide-black/[0.04]">
                {streamingEntries.map((se) => (
                  <div key={se.agentId} className="px-4 py-3 bg-violet-50/20">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      {PROVIDER_LOGOS[se.agentProvider] && <Image src={PROVIDER_LOGOS[se.agentProvider]} alt="" width={13} height={13} className="object-contain" />}
                      <span className="text-xs font-medium text-black/80">{se.agentName}</span>
                      <span className="text-[9px] px-1.5 py-0.5 border border-violet-200 text-violet-500">
                        {se.phase === "deciding" ? "reasoning" : se.phase === "thinking" ? "thinking" : se.phase === "family_discussion" ? "family" : "comm"}
                      </span>
                    </div>
                    <div className="text-xs text-black/70 whitespace-pre-wrap leading-[1.65] pl-[18px]">
                      <StreamingText text={se.text} agents={agents} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="shrink-0 border-t border-black/10 px-4 py-1.5 text-[10px] text-black/30 flex justify-between">
          <span>{rounds.length} rounds{streamingPhase && ` | ${streamingPhase}`}</span>
          <span>{Object.values(agents).filter((a) => a.alive).length}/{Object.values(agents).length} alive</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StreamEntry {
  round: number;
  type: "reasoning" | "thought" | "family" | "dm" | "broadcast";
  agentId: string;
  agentName: string;
  agentProvider: string;
  content: string;
  meta?: string;
  isSent?: boolean;
  classification?: {
    intent_tags: string[];
    moral_friction: number;
    deception_sophistication: number;
    strategic_depth: number;
    theory_of_mind: number;
    meta_awareness: number;
  };
  thinkingTokens?: number;
}

// ---------------------------------------------------------------------------
// Entry card (thoughts, family, broadcasts)
// ---------------------------------------------------------------------------

const TYPE_STYLE: Record<string, { label: string; border: string; bg: string }> = {
  reasoning: { label: "reasoning", border: "border-purple-200", bg: "bg-purple-50/30" },
  thought: { label: "thinking", border: "border-violet-200", bg: "bg-violet-50/30" },
  family: { label: "family", border: "border-blue-200", bg: "bg-blue-50/20" },
  dm: { label: "DM", border: "border-amber-200", bg: "bg-amber-50/20" },
  broadcast: { label: "broadcast", border: "border-black/10", bg: "" },
};

function ClassificationBadges({ classification, thinkingTokens }: {
  classification?: StreamEntry["classification"];
  thinkingTokens?: number;
}) {
  if (!classification && !thinkingTokens) return null;
  const badges: { label: string; color: string }[] = [];

  if (thinkingTokens) {
    badges.push({ label: `${thinkingTokens} tok`, color: "bg-purple-100 text-purple-600" });
  }

  if (classification) {
    for (const tag of classification.intent_tags.slice(0, 3)) {
      const tagColors: Record<string, string> = {
        TARGETING: "bg-red-100 text-red-600",
        DECEPTION_PLANNING: "bg-orange-100 text-orange-600",
        BETRAYAL_PLANNING: "bg-red-100 text-red-600",
        ALLIANCE_SINCERE: "bg-green-100 text-green-600",
        ALLIANCE_INSTRUMENTAL: "bg-yellow-100 text-yellow-700",
        SELF_PRESERVATION: "bg-blue-100 text-blue-600",
        THREAT_ASSESSMENT: "bg-amber-100 text-amber-600",
      };
      badges.push({ label: tag.toLowerCase().replace("_", " "), color: tagColors[tag] ?? "bg-gray-100 text-gray-500" });
    }

    if (classification.moral_friction >= 3) {
      badges.push({ label: `friction:${classification.moral_friction}`, color: "bg-emerald-100 text-emerald-600" });
    } else if (classification.moral_friction === 0 && classification.intent_tags.includes("TARGETING")) {
      badges.push({ label: "frictionless", color: "bg-red-100 text-red-600" });
    }

    if (classification.theory_of_mind >= 3) {
      badges.push({ label: `ToM:${classification.theory_of_mind}`, color: "bg-indigo-100 text-indigo-600" });
    }

    if (classification.meta_awareness >= 2) {
      badges.push({ label: `meta:${classification.meta_awareness}`, color: "bg-pink-100 text-pink-600" });
    }
  }

  if (badges.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1 mt-1.5 pl-[18px]">
      {badges.map((b, i) => (
        <span key={i} className={`text-[8px] px-1.5 py-0.5 font-medium ${b.color}`}>
          {b.label}
        </span>
      ))}
    </div>
  );
}

function EntryCard({ entry, agents, isNew }: { entry: StreamEntry; agents: Record<string, AgentState>; isNew: boolean }) {
  const style = TYPE_STYLE[entry.type] ?? TYPE_STYLE.broadcast;
  return (
    <div className={`px-4 py-3 ${style.bg}`}>
      <div className="flex items-center gap-1.5 mb-1.5">
        {PROVIDER_LOGOS[entry.agentProvider] && <Image src={PROVIDER_LOGOS[entry.agentProvider]} alt="" width={13} height={13} className="object-contain" />}
        <span className="text-xs font-medium text-black/80">{entry.agentName}</span>
        <span className={`text-[9px] px-1.5 py-0.5 ${style.border} border text-black/40`}>{style.label}</span>
        {entry.meta && <span className="text-[9px] text-black/30">{entry.meta}</span>}
      </div>
      <div className="text-xs text-black/70 whitespace-pre-wrap leading-[1.65] pl-[18px]">
        <FadeInText text={entry.content} isNew={isNew} agents={agents} />
      </div>
      {entry.type === "reasoning" && (
        <ClassificationBadges classification={entry.classification} thinkingTokens={entry.thinkingTokens} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DM bubble (chat-thread style)
// ---------------------------------------------------------------------------

function DmBubble({ entry, activeAgent, agents, isNew }: { entry: StreamEntry; activeAgent: { id: string; name: string } | null; agents: Record<string, AgentState>; isNew: boolean }) {
  const isSent = entry.isSent ?? false;
  return (
    <div className={`px-4 py-3 flex ${isSent ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] flex gap-2 ${isSent ? "flex-row-reverse" : "flex-row"}`}>
        <div className="shrink-0 mt-0.5">
          {PROVIDER_LOGOS[entry.agentProvider] ? (
            <Image src={PROVIDER_LOGOS[entry.agentProvider]} alt="" width={20} height={20} className="object-contain" />
          ) : (
            <div className="w-5 h-5 bg-black/10" />
          )}
        </div>
        <div className={`${isSent ? "bg-black/[0.06]" : "bg-amber-50/60 border border-amber-200/40"} px-3 py-2`}>
          <div className="flex items-center gap-1.5 mb-1">
            <span className="text-[11px] font-medium text-black/70">{entry.agentName}</span>
            <span className="text-[9px] text-black/30">{isSent ? `to ${entry.meta}` : `to ${activeAgent?.name ?? "you"}`}</span>
            <span className="text-[9px] text-black/20 ml-auto">R{entry.round}</span>
          </div>
          <div className="text-xs text-black/70 leading-[1.6] whitespace-pre-wrap">
            <FadeInText text={entry.content} isNew={isNew} agents={agents} />
          </div>
        </div>
      </div>
    </div>
  );
}
