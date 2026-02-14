"use client";

import { TIER_LABELS } from "@/lib/colors";

interface AgentBadgeProps {
  name: string;
  family: string;
  color: string;
  tier?: number;
  alive?: boolean;
  small?: boolean;
}

export function AgentBadge({
  name,
  color,
  tier,
  alive = true,
  small = false,
}: AgentBadgeProps) {
  const tierLabel = tier ? TIER_LABELS[tier] : undefined;

  return (
    <span
      className={`inline-flex items-center gap-1.5 ${small ? "text-xs" : "text-sm"} ${
        alive ? "" : "opacity-40 line-through"
      }`}
    >
      <span
        className="inline-block rounded-full shrink-0"
        style={{
          backgroundColor: color,
          width: small ? 8 : 10,
          height: small ? 8 : 10,
        }}
      />
      <span className="font-medium text-foreground">{name}</span>
      {tierLabel && (
        <span className="text-muted-foreground text-xs">({tierLabel})</span>
      )}
    </span>
  );
}
