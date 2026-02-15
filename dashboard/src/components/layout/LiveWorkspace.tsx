"use client";

import { useState, useMemo, useRef, useEffect, useCallback, type ReactNode } from "react";
import Image from "next/image";
import { GameGrid } from "@/components/GameGrid";
import { KillTimeline } from "@/components/KillTimeline";
import { OptimalMovesSidePanel } from "@/components/OptimalMovesPanel";
import { useGameState } from "@/hooks/useGameState";
import type { AgentState } from "@/lib/types";

const PROVIDER_LOGOS: Record<string, string> = {
  anthropic: "/logos/anthropic.png",
  openai: "/logos/openai.webp",
  google: "/logos/google.png",
  xai: "/logos/xai.png",
};

const TIER_LABELS: Record<number, string> = { 1: "Boss", 2: "Lt", 3: "Soldier" };

type MsgTab = "reasoning" | "dms";

const TAB_TO_TYPE: Record<MsgTab, string> = {
  reasoning: "reasoning", dms: "dm",
};

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

function cleanText(raw: string): string {
  let s = raw;
  s = s.replace(/```json\s*[\s\S]*?```/g, "");
  s = s.replace(/\n\s*\{[\s\S]*\}\s*$/, "");
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
          <span key={i} className="inline-flex items-center bg-blue-500/15 text-blue-600 px-1.5 py-0.5 text-[10px] font-semibold cursor-default hover:bg-blue-500/25 transition-colors" style={{ borderRadius: "9999px" }}>
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

function ThinkingShimmer({ agentName, agentModel }: { agentName: string; agentModel: string }) {
  const modelLabel = agentModel || agentName;
  return (
    <div className="flex items-center gap-2 pl-[18px]">
      <div className="h-3 flex-1 max-w-[180px] animate-shimmer" />
      <span className="text-[10px] text-black/30 italic whitespace-nowrap">{modelLabel} is thinking...</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Game timer
// ---------------------------------------------------------------------------

function GameTimer() {
  const { rounds, streamingPhase } = useGameState();
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    if (rounds.length === 0 && !streamingPhase) {
      startRef.current = null;
      setElapsed(0);
      return;
    }
    if (startRef.current === null) {
      startRef.current = Date.now();
    }
    const tick = setInterval(() => {
      if (startRef.current) setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(tick);
  }, [rounds.length, streamingPhase]);

  if (elapsed === 0 && !streamingPhase) return null;

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return (
    <span className="tabular-nums">{mins}:{secs.toString().padStart(2, "0")}</span>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function LiveWorkspace() {
  const {
    rounds, agents, families, gridSize, currentRound,
    streamingPhase, streamingTokens,
    showOptimalMoves, setShowOptimalMoves,
  } = useGameState();
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [msgTab, setMsgTab] = useState<MsgTab>("reasoning");
  const [showAllFamily, setShowAllFamily] = useState(true);
  const [isNearBottom, setIsNearBottom] = useState(true);
  const [unreadCount, setUnreadCount] = useState(0);
  const [bcastIsNearBottom, setBcastIsNearBottom] = useState(true);
  const [bcastUnreadCount, setBcastUnreadCount] = useState(0);
  const [selectedDmPartner, setSelectedDmPartner] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bcastScrollRef = useRef<HTMLDivElement>(null);
  const lastRoundCount = useRef(0);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    setIsNearBottom(nearBottom);
    if (nearBottom) setUnreadCount(0);
  }, []);

  const handleBcastScroll = useCallback(() => {
    const el = bcastScrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    setBcastIsNearBottom(nearBottom);
    if (nearBottom) setBcastUnreadCount(0);
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (isNearBottom) el.scrollTop = el.scrollHeight;
  }, [rounds.length, streamingTokens, isNearBottom]);

  useEffect(() => {
    const el = bcastScrollRef.current;
    if (!el) return;
    if (bcastIsNearBottom) el.scrollTop = el.scrollHeight;
  }, [rounds.length, bcastIsNearBottom]);

  useEffect(() => {
    if (!isNearBottom) setUnreadCount((c) => c + 1);
    if (!bcastIsNearBottom) setBcastUnreadCount((c) => c + 1);
  }, [rounds.length]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // Optimal moves: compute agent list from current round and split families
  const optAgentList = useMemo(() => {
    const roundData = currentRound > 0 ? rounds[currentRound - 1] : null;
    const source =
      roundData?.grid?.agents && roundData.grid.agents.length > 0
        ? roundData.grid.agents
        : Object.values(agents);
    return source as import("@/lib/types").AgentState[];
  }, [currentRound, rounds, agents]);

  const leftFamilies = useMemo(
    () => families.slice(0, Math.ceil(families.length / 2)).map((f) => f.name),
    [families],
  );
  const rightFamilies = useMemo(
    () => families.slice(Math.ceil(families.length / 2)).map((f) => f.name),
    [families],
  );

  // Build flat stream for right panel (reasoning + family + dms)
  const stream = useMemo(() => {
    const entries: StreamEntry[] = [];
    for (const round of rounds) {
      const rn = round.round;

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
              intent_tags: cls.intent_tags, moral_friction: cls.moral_friction,
              deception_sophistication: cls.deception_sophistication, strategic_depth: cls.strategic_depth,
              theory_of_mind: cls.theory_of_mind, meta_awareness: cls.meta_awareness,
            } : undefined,
            thinkingTokens: trace.tokens_thinking,
          });
        }
      }

      if (round.thoughts && !round.reasoning_traces) {
        for (const [agentId, thought] of Object.entries(round.thoughts)) {
          const agent = agents[agentId];
          if (!agent) continue;
          if (activeAgent && agent.id !== activeAgent.id) continue;
          if (!activeAgent && !familyAgentIds.has(agent.id)) continue;
          entries.push({ round: rn, type: "reasoning", agentId, agentName: agent.name, agentProvider: agent.provider, content: thought });
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
        const isRelevant = activeAgent
          ? dm.sender === activeAgent.id || (dm.recipient?.toLowerCase() === activeAgent.name.toLowerCase())
          : familyAgentIds.has(dm.sender) || familyAgents.some((a) => dm.recipient?.toLowerCase() === a.name.toLowerCase());
        if (!isRelevant) continue;
        const senderAgent = Object.values(agents).find((a) => a.id === dm.sender);
        const isSent = activeAgent ? dm.sender === activeAgent.id : familyAgentIds.has(dm.sender);
        entries.push({ round: rn, type: "dm", agentId: dm.sender, agentName: dm.sender_name, agentProvider: senderAgent?.provider ?? "", content: dm.content, meta: dm.recipient ?? "", isSent });
      }
    }
    return entries;
  }, [rounds, agents, activeAgent, activeFamily, familyAgentIds, familyAgents, provider]);

  // Build broadcast stream (all agents, all rounds)
  const broadcastStream = useMemo(() => {
    const entries: StreamEntry[] = [];
    for (const round of rounds) {
      const bcast = round.messages?.broadcasts ?? [];
      for (const msg of bcast) {
        const senderAgent = Object.values(agents).find((a) => a.id === msg.sender);
        entries.push({
          round: round.round, type: "broadcast", agentId: msg.sender,
          agentName: msg.sender_name, agentProvider: senderAgent?.provider ?? "",
          content: msg.content,
        });
      }
    }
    return entries;
  }, [rounds, agents]);

  const filtered = showAllFamily && !selectedAgent
    ? stream.filter((e) => e.type === "family")
    : stream.filter((e) => e.type === TAB_TO_TYPE[msgTab]);

  const byRound = useMemo(() => {
    const map = new Map<number, StreamEntry[]>();
    for (const e of filtered) {
      (map.get(e.round) ?? (() => { const a: StreamEntry[] = []; map.set(e.round, a); return a; })()).push(e);
    }
    return Array.from(map.entries()).sort((a, b) => a[0] - b[0]);
  }, [filtered]);

  // DM inbox: group by conversation partner
  const dmInbox = useMemo(() => {
    const dmEntries = stream.filter((e) => e.type === "dm");
    const threads: Record<string, { partnerName: string; partnerProvider: string; messages: StreamEntry[]; lastRound: number }> = {};
    for (const dm of dmEntries) {
      const isSent = dm.isSent ?? false;
      const partnerName = isSent ? (dm.meta ?? dm.agentName) : dm.agentName;
      const partnerProvider = isSent ? "" : dm.agentProvider;
      const key = partnerName.toLowerCase();
      if (!threads[key]) {
        threads[key] = { partnerName, partnerProvider: partnerProvider || findProviderByName(partnerName, agents), messages: [], lastRound: 0 };
      }
      threads[key].messages.push(dm);
      threads[key].lastRound = Math.max(threads[key].lastRound, dm.round);
      if (!threads[key].partnerProvider && dm.agentProvider) threads[key].partnerProvider = dm.agentProvider;
    }
    return Object.values(threads).sort((a, b) => b.lastRound - a.lastRound);
  }, [stream, agents]);

  const streamingEntries = useMemo(() => {
    if (!streamingPhase) return [];
    const relevantAgents = activeAgent
      ? [activeAgent].filter((a) => a.alive)
      : familyAgents.filter((a) => a.alive);
    return relevantAgents.map((agent) => ({
      agentId: agent.id, agentName: agent.name,
      agentProvider: agent.provider, agentModel: agent.model,
      text: streamingTokens[agent.id] || "", phase: streamingPhase,
    }));
  }, [streamingPhase, streamingTokens, activeAgent, familyAgents]);

  // Filter streaming entries by current view context
  const visibleStreamingEntries = useMemo(() => {
    if (showAllFamily && !selectedAgent) {
      // Family (All) view: only show family_discussion phase
      return streamingEntries.filter((se) => se.phase === "family_discussion");
    }
    if (selectedAgent && msgTab === "dms") {
      // DMs view: no streaming
      return [];
    }
    if (selectedAgent && msgTab === "reasoning") {
      // Reasoning [Private] view: only deciding/thinking phase
      return streamingEntries.filter((se) => se.phase === "deciding" || se.phase === "thinking");
    }
    return streamingEntries;
  }, [streamingEntries, showAllFamily, selectedAgent, msgTab]);

  return (
    <div className="flex-1 min-h-0 flex">
      {/* LEFT: Grid */}
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="shrink-0 flex items-center justify-between px-4 h-8 border-b border-black/5">
          <span className="text-[10px] text-black/30 font-medium">
            {rounds.length > 0 ? `Round ${rounds[rounds.length - 1]?.round ?? 0}` : "Waiting..."}
          </span>
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showOptimalMoves}
              onChange={(e) => setShowOptimalMoves(e.target.checked)}
              className="w-3 h-3 cursor-pointer"
            />
            <span className={`text-[10px] ${showOptimalMoves ? "text-black/50 font-medium" : "text-black/30"}`}>
              Optimal moves
            </span>
          </label>
          <span className="text-[10px] text-black/30 font-mono"><GameTimer /></span>
        </div>
        <div className="flex-1 min-h-0 flex">
          {showOptimalMoves && (
            <OptimalMovesSidePanel
              familyNames={leftFamilies}
              allAgents={optAgentList}
              families={families}
              gridSize={gridSize}
              side="left"
            />
          )}
          <div className="flex-1 min-w-0"><GameGrid /></div>
          {showOptimalMoves && (
            <OptimalMovesSidePanel
              familyNames={rightFamilies}
              allAgents={optAgentList}
              families={families}
              gridSize={gridSize}
              side="right"
            />
          )}
        </div>
        <div className="h-[80px] shrink-0 border-t border-black/5"><KillTimeline /></div>
      </div>

      {/* MIDDLE: Broadcast feed */}
      <div className="w-[260px] shrink-0 flex flex-col h-full overflow-hidden border-l border-black/10">
        <div className="shrink-0 px-4 py-2 text-[11px] font-medium text-black/60 border-b border-black/10">Broadcasts</div>
        <div ref={bcastScrollRef} onScroll={handleBcastScroll} className="flex-1 min-h-0 overflow-y-auto relative">
          {broadcastStream.length === 0 && (
            <div className="text-xs text-black/25 text-center py-8">No broadcasts yet</div>
          )}
          {broadcastStream.map((entry, i) => (
            <div key={`bcast-${i}`} className="px-3 py-2.5 border-b border-black/[0.04]">
              <div className="flex items-center gap-1.5 mb-1">
                {PROVIDER_LOGOS[entry.agentProvider] && <Image src={PROVIDER_LOGOS[entry.agentProvider]} alt="" width={12} height={12} className="object-contain" />}
                <span className="text-[11px] font-medium text-black/70">{entry.agentName}</span>
                <span className="text-[9px] text-black/25 ml-auto">R{entry.round}</span>
              </div>
              <div className="text-[11px] text-black/60 leading-[1.55] whitespace-pre-wrap">{cleanText(entry.content)}</div>
            </div>
          ))}
          {!bcastIsNearBottom && bcastUnreadCount > 0 && (
            <button
              onClick={() => { const el = bcastScrollRef.current; if (el) el.scrollTop = el.scrollHeight; setBcastUnreadCount(0); }}
              className="sticky bottom-2 left-1/2 -translate-x-1/2 z-20 bg-black text-white text-[10px] px-3 py-1.5 shadow-lg hover:bg-black/80 transition-colors"
            >
              {bcastUnreadCount} new
            </button>
          )}
        </div>
        <div className="shrink-0 border-t border-black/10 px-3 py-1.5 text-[9px] text-black/25">{broadcastStream.length} broadcasts</div>
      </div>

      {/* RIGHT: Stream panel */}
      <div className="w-[400px] shrink-0 flex flex-col h-full overflow-hidden border-l border-black/10">
        {/* Family tabs */}
        <div className="flex shrink-0">
          {families.map((fam) => (
            <button key={fam.name} onClick={() => { setSelectedFamily(fam.name); setSelectedAgent(null); setSelectedDmPartner(null); }}
              className={`flex-1 flex items-center justify-center gap-1.5 px-1 py-2 text-[11px] font-medium border-b-2 transition-colors ${activeFamily === fam.name ? "border-black text-black" : "border-transparent text-black/35 hover:text-black/60"}`}>
              {PROVIDER_LOGOS[fam.provider] && <Image src={PROVIDER_LOGOS[fam.provider]} alt="" width={14} height={14} className="object-contain" />}
              {fam.name}
            </button>
          ))}
        </div>

        {/* Row 2: Family (All) | individual agents */}
        <div className="flex shrink-0 bg-black/[0.02] border-b border-black/10">
          <button onClick={() => { setSelectedAgent(null); setShowAllFamily(true); setSelectedDmPartner(null); }}
            className={`flex-1 px-1 py-1 text-[10px] border-b-2 transition-colors ${showAllFamily && !selectedAgent ? "border-black text-black font-medium" : "border-transparent text-black/40"}`}>
            Family (All)
          </button>
          {familyAgents.sort((a, b) => a.tier - b.tier).map((agent) => (
            <button key={agent.id} onClick={() => { setSelectedAgent(agent.id); setShowAllFamily(false); setSelectedDmPartner(null); }}
              className={`flex-1 px-1 py-1 text-[10px] border-b-2 transition-colors ${activeAgent?.id === agent.id ? "border-black text-black font-medium" : "border-transparent text-black/40"}`}>
              {agent.name}
              {!agent.alive && <span className="text-red-400 ml-0.5">X</span>}
            </button>
          ))}
        </div>

        {/* Row 3: Reasoning [Private] | DMs — only when individual agent selected */}
        {!showAllFamily && selectedAgent && (
          <div className="flex shrink-0 border-b border-black/10">
            {(["reasoning", "dms"] as MsgTab[]).map((tab) => (
              <button key={tab} onClick={() => { setMsgTab(tab); setSelectedDmPartner(null); }}
                className={`flex-1 px-1 py-1.5 text-[10px] border-b-2 transition-colors ${msgTab === tab ? "border-black text-black font-medium" : "border-transparent text-black/35"}`}>
                {tab === "reasoning" ? "Reasoning [Private]" : "DMs"}
              </button>
            ))}
          </div>
        )}

        {/* Scrollable stream */}
        <div ref={scrollRef} onScroll={handleScroll} className="flex-1 min-h-0 overflow-y-auto relative">
          {rounds.length === 0 && streamingEntries.length === 0 && (
            <div className="text-xs text-black/30 text-center py-12">Waiting for game data...</div>
          )}

          {/* DM inbox view */}
          {!showAllFamily && msgTab === "dms" && !selectedDmPartner && (
            <div>
              {dmInbox.length === 0 && rounds.length > 0 && (
                <div className="text-xs text-black/25 text-center py-8">No DMs yet</div>
              )}
              {dmInbox.map((thread) => {
                const lastMsg = thread.messages[thread.messages.length - 1];
                const threadProvider = thread.partnerProvider || findProviderByName(thread.partnerName, agents);
                return (
                  <button
                    key={thread.partnerName}
                    onClick={() => setSelectedDmPartner(thread.partnerName.toLowerCase())}
                    className="w-full px-4 py-3 flex items-start gap-3 border-b border-black/[0.05] hover:bg-black/[0.02] transition-colors text-left"
                  >
                    <div className="shrink-0 mt-0.5">
                      {PROVIDER_LOGOS[threadProvider] ? (
                        <Image src={PROVIDER_LOGOS[threadProvider]} alt="" width={28} height={28} className="object-contain" />
                      ) : (
                        <div className="w-7 h-7 bg-black/10" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-[12px] font-semibold text-black/80">{thread.partnerName}</span>
                        <span className="text-[9px] text-black/30">R{thread.lastRound}</span>
                      </div>
                      <div className="text-[11px] text-black/45 truncate leading-snug">
                        {lastMsg?.isSent ? "You: " : ""}{cleanText(lastMsg?.content ?? "").slice(0, 80)}
                      </div>
                    </div>
                    <div className="shrink-0 mt-1">
                      <span className="text-[9px] bg-black/[0.06] text-black/40 px-1.5 py-0.5 font-medium">{thread.messages.length}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {/* DM thread view */}
          {!showAllFamily && msgTab === "dms" && selectedDmPartner && (
            <div>
              <button
                onClick={() => setSelectedDmPartner(null)}
                className="sticky top-0 z-10 w-full bg-[#f7f7f7] px-4 py-1.5 text-[10px] font-medium text-black/50 border-b border-black/5 flex items-center gap-1 hover:text-black/70"
              >
                &larr; Back to inbox
              </button>
              {filtered
                .filter((e) => {
                  const partnerName = e.isSent ? (e.meta ?? "").toLowerCase() : e.agentName.toLowerCase();
                  return partnerName === selectedDmPartner;
                })
                .map((entry, i) => (
                  <DmBubble key={`dm-thread-${i}`} entry={entry} activeAgent={activeAgent} agents={agents} isNew={entry.round === newRoundNum} />
                ))
              }
            </div>
          )}

          {/* Standard stream view (family-all or reasoning) */}
          {(showAllFamily || msgTab !== "dms") && byRound.map(([roundNum, entries]) => (
            <div key={roundNum}>
              <div className="sticky top-0 z-10 bg-[#f7f7f7] px-4 py-1.5 text-[10px] font-medium text-black/50 border-b border-black/5">
                Round {roundNum}
              </div>
              <div className="divide-y divide-black/[0.04]">
                {entries.map((entry, i) => (
                  <EntryCard key={`${roundNum}-${entry.type}-${entry.agentId}-${i}`} entry={entry} agents={agents} isNew={roundNum === newRoundNum} />
                ))}
              </div>
            </div>
          ))}

          {/* Live streaming section */}
          {visibleStreamingEntries.length > 0 && (
            <div>
              <div className="sticky top-0 z-10 bg-[#f7f7f7] px-4 py-1.5 text-[10px] font-medium text-black/50 border-b border-black/5 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-black/40 animate-pulse" />
                {streamingPhase === "deciding" ? "Reasoning..." : streamingPhase === "thinking" ? "Thinking..." : streamingPhase === "family_discussion" ? "Family Discussion..." : "Communicating..."}
              </div>
              <div className="divide-y divide-black/[0.04]">
                {visibleStreamingEntries.map((se) => (
                  <div key={se.agentId} className="px-4 py-3">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      {PROVIDER_LOGOS[se.agentProvider] && <Image src={PROVIDER_LOGOS[se.agentProvider]} alt="" width={13} height={13} className="object-contain" />}
                      <span className="text-xs font-medium text-black/80">{se.agentName}</span>
                      <span className="text-[9px] px-1.5 py-0.5 border border-black/10 text-black/40">
                        {se.phase === "deciding" ? "reasoning" : se.phase === "thinking" ? "thinking" : se.phase === "family_discussion" ? "family" : "comm"}
                      </span>
                    </div>
                    {se.text ? (
                      <div className="text-xs text-black/70 whitespace-pre-wrap leading-[1.65] pl-[18px]">
                        <StreamingText text={se.text} agents={agents} />
                      </div>
                    ) : (
                      <ThinkingShimmer agentName={se.agentName} agentModel={se.agentModel} />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Unread messages pill */}
          {!isNearBottom && unreadCount > 0 && (
            <button
              onClick={() => { const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight; setUnreadCount(0); }}
              className="sticky bottom-2 left-1/2 -translate-x-1/2 z-20 bg-black text-white text-[10px] px-3 py-1.5 shadow-lg hover:bg-black/80 transition-colors"
            >
              {unreadCount} new {unreadCount === 1 ? "message" : "messages"} below
            </button>
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
// Helpers
// ---------------------------------------------------------------------------

function findProviderByName(name: string, agents: Record<string, AgentState>): string {
  const agent = Object.values(agents).find((a) => a.name.toLowerCase() === name.toLowerCase());
  return agent?.provider ?? "";
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StreamEntry {
  round: number;
  type: "reasoning" | "family" | "dm" | "broadcast";
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
// Entry card (reasoning, family)
// ---------------------------------------------------------------------------

const TYPE_STYLE: Record<string, { label: string; border: string; bg: string }> = {
  reasoning: { label: "reasoning", border: "border-black/15", bg: "bg-black/[0.02]" },
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
    badges.push({ label: `${thinkingTokens} tok`, color: "bg-black/[0.06] text-black/50" });
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
