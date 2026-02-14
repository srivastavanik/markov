"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { useGameState } from "@/hooks/useGameState";
import { getFamilyColor } from "@/lib/colors";

export function DeceptionChart() {
  const { rounds, currentRound, agents } = useGameState();

  const visibleRounds = rounds.slice(0, currentRound);

  const { chartData, agentIds, chartConfig } = useMemo(() => {
    const ids = Object.keys(agents);
    const data = visibleRounds.map((r) => {
      const point: Record<string, number | string> = { round: r.round };
      for (const id of ids) {
        const delta = r.analysis?.[id]?.deception_delta;
        if (delta !== undefined) {
          point[id] = Math.round(delta * 100) / 100;
        }
      }
      return point;
    });

    const config: Record<string, { label: string; color: string }> = {};
    for (const id of ids) {
      const agent = agents[id];
      if (agent) {
        config[id] = {
          label: agent.name,
          color: getFamilyColor(agent.family),
        };
      }
    }

    return { chartData: data, agentIds: ids, chartConfig: config };
  }, [visibleRounds, agents, currentRound]);

  if (chartData.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm font-semibold">Deception Delta</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-32">
          <p className="text-sm text-muted-foreground">No data yet</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full">
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm font-semibold">Deception Delta</CardTitle>
      </CardHeader>
      <CardContent className="p-2">
        <ChartContainer config={chartConfig} className="h-[180px] w-full">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis
              dataKey="round"
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={30}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            {agentIds.map((id) => {
              const agent = agents[id];
              if (!agent) return null;
              return (
                <Line
                  key={id}
                  type="monotone"
                  dataKey={id}
                  stroke={getFamilyColor(agent.family)}
                  strokeWidth={agent.alive ? 2 : 1}
                  strokeOpacity={agent.alive ? 1 : 0.3}
                  dot={false}
                  name={agent.name}
                />
              );
            })}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
