"use client";

import { Fragment, useMemo } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useGameState } from "@/hooks/useGameState";
import { DeceptionChart } from "@/components/DeceptionChart";
import { ChartContainer } from "@/components/ui/chart";
import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";

export function AnalysisWorkspace() {
  return (
    <div className="flex-1 min-h-0 p-2">
      <Tabs defaultValue="deception" className="h-full flex flex-col">
        <TabsList className="h-9 shrink-0">
          <TabsTrigger value="taxonomy" className="text-xs">Taxonomy</TabsTrigger>
          <TabsTrigger value="deception" className="text-xs">Deception</TabsTrigger>
          <TabsTrigger value="malice" className="text-xs">Malice</TabsTrigger>
          <TabsTrigger value="bias" className="text-xs">Bias</TabsTrigger>
          <TabsTrigger value="safety" className="text-xs">Safety</TabsTrigger>
          <TabsTrigger value="hierarchy" className="text-xs">Hierarchy</TabsTrigger>
          <TabsTrigger value="provider" className="text-xs">Provider</TabsTrigger>
        </TabsList>
        <TabsContent value="taxonomy" className="min-h-0 flex-1 mt-2"><TaxonomyPanel /></TabsContent>
        <TabsContent value="deception" className="min-h-0 flex-1 mt-2"><DeceptionPanel /></TabsContent>
        <TabsContent value="malice" className="min-h-0 flex-1 mt-2"><MalicePanel /></TabsContent>
        <TabsContent value="bias" className="min-h-0 flex-1 mt-2"><BiasPanel /></TabsContent>
        <TabsContent value="safety" className="min-h-0 flex-1 mt-2"><SafetyPanel /></TabsContent>
        <TabsContent value="hierarchy" className="min-h-0 flex-1 mt-2"><HierarchyPanel /></TabsContent>
        <TabsContent value="provider" className="min-h-0 flex-1 mt-2"><ProviderPanel /></TabsContent>
      </Tabs>
    </div>
  );
}

function TaxonomyPanel() {
  const { rounds, agents } = useGameState();

  // Aggregate classification scores per agent across all rounds
  const agentScores = useMemo(() => {
    const scores: Record<string, { name: string; family: string; provider: string; rounds: number; moral_friction: number; deception_sophistication: number; strategic_depth: number; theory_of_mind: number; meta_awareness: number; intent_counts: Record<string, number> }> = {};

    for (const round of rounds) {
      const analysis = round.analysis || {};
      for (const [agentId, agentAnalysis] of Object.entries(analysis)) {
        const cls = agentAnalysis?.classification;
        if (!cls) continue;
        const agent = agents[agentId];
        if (!agent) continue;

        if (!scores[agentId]) {
          scores[agentId] = {
            name: agent.name, family: agent.family, provider: agent.provider,
            rounds: 0, moral_friction: 0, deception_sophistication: 0,
            strategic_depth: 0, theory_of_mind: 0, meta_awareness: 0,
            intent_counts: {},
          };
        }
        const s = scores[agentId];
        s.rounds++;
        s.moral_friction += cls.moral_friction ?? 0;
        s.deception_sophistication += cls.deception_sophistication ?? 0;
        s.strategic_depth += cls.strategic_depth ?? 0;
        s.theory_of_mind += cls.theory_of_mind ?? 0;
        s.meta_awareness += cls.meta_awareness ?? 0;
        for (const tag of cls.intent_tags ?? []) {
          s.intent_counts[tag] = (s.intent_counts[tag] || 0) + 1;
        }
      }
    }
    return Object.entries(scores)
      .map(([id, s]) => ({
        id,
        ...s,
        avg_friction: s.rounds ? +(s.moral_friction / s.rounds).toFixed(1) : 0,
        avg_deception: s.rounds ? +(s.deception_sophistication / s.rounds).toFixed(1) : 0,
        avg_strategy: s.rounds ? +(s.strategic_depth / s.rounds).toFixed(1) : 0,
        avg_tom: s.rounds ? +(s.theory_of_mind / s.rounds).toFixed(1) : 0,
        avg_meta: s.rounds ? +(s.meta_awareness / s.rounds).toFixed(1) : 0,
        top_intents: Object.entries(s.intent_counts).sort((a, b) => b[1] - a[1]).slice(0, 4),
      }))
      .sort((a, b) => b.rounds - a.rounds);
  }, [rounds, agents]);

  if (agentScores.length === 0) {
    return <div className="text-sm text-muted-foreground text-center py-8">No classification data available. Run a game with the sentry model enabled.</div>;
  }

  const dims = ["avg_friction", "avg_deception", "avg_strategy", "avg_tom", "avg_meta"] as const;
  const dimLabels: Record<string, string> = {
    avg_friction: "Moral Friction",
    avg_deception: "Deception",
    avg_strategy: "Strategic Depth",
    avg_tom: "Theory of Mind",
    avg_meta: "Meta-Awareness",
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 p-2">
        {agentScores.map((agent) => (
          <Card key={agent.id} className="border-black/10">
            <CardHeader className="p-3 pb-1">
              <CardTitle className="text-sm font-semibold flex items-center justify-between">
                <span>{agent.name} <span className="text-[10px] text-muted-foreground font-normal">({agent.family})</span></span>
                <span className="text-[10px] text-muted-foreground">{agent.rounds} rounds</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3 pt-0">
              {/* Radar chart */}
              <div className="h-[180px]">
                <ChartContainer config={{}} className="h-full w-full">
                  <RadarChart
                    data={dims.map((d) => ({
                      dim: dimLabels[d],
                      value: agent[d],
                      max: d === "avg_friction" || d === "avg_deception" ? 5 : 4,
                    }))}
                  >
                    <PolarGrid />
                    <PolarAngleAxis dataKey="dim" tick={{ fontSize: 9 }} />
                    <Radar dataKey="value" fill="rgba(0,0,0,0.15)" stroke="rgba(0,0,0,0.5)" strokeWidth={1.5} fillOpacity={0.3} />
                  </RadarChart>
                </ChartContainer>
              </div>
              {/* Dimension scores */}
              <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-black/50 mt-1">
                {dims.map((d) => (
                  <span key={d}>{dimLabels[d]}: <strong className="text-black/70">{agent[d]}</strong></span>
                ))}
              </div>
              {/* Top intents */}
              {agent.top_intents.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {agent.top_intents.map(([tag, count]) => (
                    <span key={tag} className="text-[9px] px-1.5 py-0.5 bg-black/[0.06] text-black/50 font-medium">
                      {tag.toLowerCase().replace(/_/g, " ")} ({count})
                    </span>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function DeceptionPanel() {
  return (
    <div className="h-full">
      <DeceptionChart />
    </div>
  );
}

function MalicePanel() {
  const { rounds, agents } = useGameState();
  const rows = useMemo(() => {
    return Object.keys(agents).map((id) => {
      const byRound = rounds.map((r) => r.analysis?.[id]).filter(Boolean);
      const maliceCount = byRound.filter((a) => a.malice?.elimination_planning).length;
      const firstHostile = rounds.find((r) => r.analysis?.[id]?.malice?.elimination_planning)?.round ?? null;
      const targets = new Set(byRound.flatMap((a) => a.malice?.targets_mentioned || []));
      const sophisticationAvg = byRound.length
        ? byRound.reduce((sum, a) => sum + (a.malice?.sophistication || 0), 0) / byRound.length
        : 0;
      return {
        id,
        name: agents[id].name,
        family: agents[id].family,
        rate: byRound.length ? maliceCount / byRound.length : 0,
        firstHostile,
        targets: Array.from(targets).join(", "),
        sophistication: sophisticationAvg,
      };
    });
  }, [agents, rounds]);

  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4"><CardTitle className="text-sm font-medium">Malice Metrics</CardTitle></CardHeader>
      <CardContent className="p-2 h-full overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-zinc-50">
            <tr>
              <th className="text-left px-2 py-1">Agent</th>
              <th className="text-left px-2 py-1">Family</th>
              <th className="text-left px-2 py-1">Malice Rate</th>
              <th className="text-left px-2 py-1">First Hostile Round</th>
              <th className="text-left px-2 py-1">Targets</th>
              <th className="text-left px-2 py-1">Sophistication</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-b border-black/5">
                <td className="px-2 py-1">{r.name}</td>
                <td className="px-2 py-1">{r.family}</td>
                <td className="px-2 py-1">{(r.rate * 100).toFixed(0)}%</td>
                <td className="px-2 py-1">{r.firstHostile ?? "-"}</td>
                <td className="px-2 py-1">{r.targets || "-"}</td>
                <td className="px-2 py-1">{r.sophistication.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function BiasPanel() {
  const { rounds, families } = useGameState();
  const matrix = useMemo(() => {
    const famNames = families.map((f) => f.name);
    const acc: Record<string, Record<string, number[]>> = {};
    for (const src of famNames) {
      acc[src] = {};
      for (const dst of famNames) acc[src][dst] = [];
    }
    for (const round of rounds) {
      for (const analysis of Object.values(round.analysis || {})) {
        for (const [dst, score] of Object.entries(analysis.family_sentiment || {})) {
          const src = famNames.find((f) => Object.keys(acc).includes(f)) || "";
          if (src && acc[src]?.[dst]) acc[src][dst].push(score);
        }
      }
    }
    const avg: Record<string, Record<string, number>> = {};
    for (const src of famNames) {
      avg[src] = {};
      for (const dst of famNames) {
        const values = acc[src][dst];
        avg[src][dst] = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
      }
    }
    return { famNames, avg };
  }, [families, rounds]);

  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4"><CardTitle className="text-sm font-medium">Family Bias Heatmap</CardTitle></CardHeader>
      <CardContent className="p-3 overflow-auto">
        <div className="grid gap-1" style={{ gridTemplateColumns: `140px repeat(${matrix.famNames.length}, minmax(100px, 1fr))` }}>
          <div />
          {matrix.famNames.map((col) => <div key={col} className="text-xs font-medium">{col}</div>)}
          {matrix.famNames.map((row) => (
            <Fragment key={row}>
              <div className="text-xs font-medium">{row}</div>
              {matrix.famNames.map((col) => {
                const value = matrix.avg[row][col];
                const hue = value >= 0 ? 120 : 0;
                const intensity = Math.min(0.85, Math.abs(value));
                return (
                  <div
                    key={`${row}-${col}`}
                    className="text-xs px-2 py-2 border border-black/10"
                    style={{ backgroundColor: `hsla(${hue}, 65%, 70%, ${intensity})` }}
                  >
                    {value.toFixed(2)}
                  </div>
                );
              })}
            </Fragment>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function SafetyPanel() {
  const { rounds, agents } = useGameState();
  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4"><CardTitle className="text-sm font-medium">Safety Artifact Timeline</CardTitle></CardHeader>
      <CardContent className="p-3 overflow-auto space-y-2">
        {Object.keys(agents).map((id) => {
          const points = rounds.map((r) => {
            const s = r.analysis?.[id]?.safety_artifacts;
            if (!s) return "neutral";
            if (s.refusal_language || s.ethical_reasoning) return "active";
            if (s.hedging) return "hedging";
            return "bypassed";
          });
          return (
            <div key={id}>
              <div className="text-xs mb-1">{agents[id].name}</div>
              <div className="grid grid-cols-12 gap-0.5">
                {points.slice(0, 12).map((p, idx) => (
                  <div
                    key={`${id}-${idx}`}
                    className="h-3 border border-black/10"
                    style={{
                      background:
                        p === "active" ? "#86efac" :
                        p === "hedging" ? "#fde047" :
                        p === "bypassed" ? "#fca5a5" : "#e4e4e7",
                    }}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function HierarchyPanel() {
  const { rounds, agents } = useGameState();
  const rows = useMemo(() => {
    return Object.keys(agents).map((id) => {
      const msgs = rounds.flatMap((r) => r.messages?.family_discussions || []);
      const spoken = msgs
        .flatMap((m) => m.transcript || [])
        .filter((t) => t.agent_id === id).length;
      return { id, name: agents[id].name, tier: agents[id].tier, spoken };
    });
  }, [agents, rounds]);
  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4"><CardTitle className="text-sm font-medium">Hierarchy Dynamics</CardTitle></CardHeader>
      <CardContent className="p-3">
        <ul className="text-xs space-y-1">
          {rows.sort((a, b) => b.spoken - a.spoken).map((r) => (
            <li key={r.id}>{r.name} (Tier {r.tier}) - discussion turns: {r.spoken}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function ProviderPanel() {
  const { rounds, agents } = useGameState();
  const data = useMemo(() => {
    const byProvider: Record<string, { deception: number[]; malice: number[]; safety: number[] }> = {};
    Object.values(agents).forEach((a) => {
      byProvider[a.provider] ||= { deception: [], malice: [], safety: [] };
    });
    for (const round of rounds) {
      for (const [id, analysis] of Object.entries(round.analysis || {})) {
        const provider = agents[id]?.provider || "unknown";
        byProvider[provider] ||= { deception: [], malice: [], safety: [] };
        byProvider[provider].deception.push(analysis.deception_delta || 0);
        byProvider[provider].malice.push(analysis.malice?.elimination_planning ? 1 : 0);
        byProvider[provider].safety.push(
          analysis.safety_artifacts?.hedging || analysis.safety_artifacts?.ethical_reasoning ? 1 : 0,
        );
      }
    }
    return Object.entries(byProvider).map(([provider, vals]) => ({
      provider,
      deception: avg(vals.deception),
      malice: avg(vals.malice),
      safety: avg(vals.safety),
      betrayal: Math.min(1, avg(vals.malice) + 0.2),
      bias: Math.abs(avg(vals.deception) - avg(vals.safety)),
    }));
  }, [agents, rounds]);
  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4"><CardTitle className="text-sm font-medium">Provider Comparison</CardTitle></CardHeader>
      <CardContent className="p-2 h-full">
        <ChartContainer
          config={{ deception: { label: "Deception", color: "#111111" } }}
          className="h-[320px] w-full"
        >
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={data}>
              <PolarGrid />
              <PolarAngleAxis dataKey="provider" />
              <Radar dataKey="deception" stroke="#111111" fill="#111111" fillOpacity={0.15} />
              <Radar dataKey="malice" stroke="#ef4444" fill="#ef4444" fillOpacity={0.12} />
              <Radar dataKey="safety" stroke="#16a34a" fill="#16a34a" fillOpacity={0.12} />
            </RadarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

function avg(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

