"use client";

import { useRef, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useGameState } from "@/hooks/useGameState";
import { getFamilyColor, SENTIMENT_COLORS, TIER_SIZES } from "@/lib/colors";
import * as d3 from "d3";

interface Node {
  id: string;
  name: string;
  family: string;
  color: string;
  tier: number;
  alive: boolean;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface Link {
  source: string;
  target: string;
  sentiment: number; // -1 to 1
  deceptive: boolean;
}

export function RelationshipWeb() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { currentRound, rounds, agents } = useGameState();

  const roundData = currentRound > 0 ? rounds[currentRound - 1] : null;

  const { nodes, links } = useMemo(() => {
    const nodeList: Node[] = Object.values(agents).map((a) => ({
      id: a.id,
      name: a.name,
      family: a.family,
      color: getFamilyColor(a.family),
      tier: a.tier,
      alive: a.alive,
    }));

    const linkList: Link[] = [];

    if (roundData?.analysis) {
      for (const [agentId, analysis] of Object.entries(roundData.analysis)) {
        const familySentiment = analysis.family_sentiment || {};
        for (const [familyName, score] of Object.entries(familySentiment)) {
          if (Math.abs(score) < 0.15) continue;

          // Find an agent in that family to link to
          const targetAgent = nodeList.find(
            (n) => n.family === familyName && n.id !== agentId && n.alive
          );
          if (!targetAgent) continue;

          // Check for deception (thought negative, message positive or vice versa)
          const deception = analysis.deception_delta > 0.5;

          linkList.push({
            source: agentId,
            target: targetAgent.id,
            sentiment: score,
            deceptive: deception && score < 0,
          });
        }
      }
    }

    return { nodes: nodeList, links: linkList };
  }, [agents, roundData, currentRound]);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = 280;
    const height = 280;

    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g");

    // Create simulation
    const sim = d3
      .forceSimulation<Node>(nodes)
      .force(
        "link",
        d3
          .forceLink<Node, Link>(links)
          .id((d) => d.id)
          .distance(80)
      )
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(20));

    // Links
    const linkElements = g
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", (d) => {
        if (d.deceptive) return SENTIMENT_COLORS.deceptive;
        return d.sentiment > 0
          ? SENTIMENT_COLORS.positive
          : SENTIMENT_COLORS.negative;
      })
      .attr("stroke-width", (d) => Math.min(Math.abs(d.sentiment) * 3, 3))
      .attr("stroke-opacity", 0.5)
      .attr("stroke-dasharray", (d) => (d.deceptive ? "4,3" : "none"));

    // Node groups
    const nodeGroups = g
      .selectAll<SVGGElement, Node>("g.node")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", "node");

    // Circle for each node
    nodeGroups
      .append("circle")
      .attr("r", (d) => {
        if (!d.alive) return 4;
        return (TIER_SIZES[d.tier] || 16) / 2.5;
      })
      .attr("fill", (d) => (d.alive ? d.color : "#D1D5DB"))
      .attr("stroke", (d) => (d.alive ? "#00000020" : "none"))
      .attr("stroke-width", 1);

    // Labels
    nodeGroups
      .append("text")
      .text((d) => d.name)
      .attr("text-anchor", "middle")
      .attr("dy", (d) => ((TIER_SIZES[d.tier] || 16) / 2.5) + 12)
      .attr("font-size", 9)
      .attr("font-family", "Inter, sans-serif")
      .attr("fill", (d) => (d.alive ? "#374151" : "#9CA3AF"));

    sim.on("tick", () => {
      linkElements
        .attr("x1", (d) => (d.source as unknown as Node).x || 0)
        .attr("y1", (d) => (d.source as unknown as Node).y || 0)
        .attr("x2", (d) => (d.target as unknown as Node).x || 0)
        .attr("y2", (d) => (d.target as unknown as Node).y || 0);

      nodeGroups.attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
    });

    return () => {
      sim.stop();
    };
  }, [nodes, links]);

  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm font-semibold">Relationships</CardTitle>
      </CardHeader>
      <CardContent className="flex items-center justify-center p-2">
        <svg ref={svgRef} className="w-full max-h-[300px]" />
      </CardContent>
      <div className="px-4 pb-3 flex gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 inline-block" style={{ backgroundColor: SENTIMENT_COLORS.positive }} /> Trust
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 inline-block" style={{ backgroundColor: SENTIMENT_COLORS.negative }} /> Hostile
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 inline-block border-dashed border-t" style={{ borderColor: SENTIMENT_COLORS.deceptive }} /> Deceptive
        </span>
      </div>
    </Card>
  );
}
