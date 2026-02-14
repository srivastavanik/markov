// Family colors and tier labels

export const FAMILY_COLORS: Record<string, string> = {
  "House Clair": "#7C6BFF",
  "House Syne": "#4ADE80",
  "House Lux": "#FACC15",
  "House Vex": "#F87171",
};

export const TIER_LABELS: Record<number, string> = {
  1: "Boss",
  2: "Lieutenant",
  3: "Soldier",
};

export const TIER_SIZES: Record<number, number> = {
  1: 28,
  2: 22,
  3: 16,
};

export const SEVERITY_COLORS: Record<string, string> = {
  critical: "#DC2626",
  high: "#EA580C",
  medium: "#CA8A04",
};

export const SENTIMENT_COLORS = {
  positive: "#16A34A",
  negative: "#DC2626",
  deceptive: "#D97706",
  neutral: "#9CA3AF",
};

export function getFamilyColor(familyName: string): string {
  return FAMILY_COLORS[familyName] || "#9CA3AF";
}

export function lightenColor(hex: string, amount: number = 0.85): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const lr = Math.round(r + (255 - r) * amount);
  const lg = Math.round(g + (255 - g) * amount);
  const lb = Math.round(b + (255 - b) * amount);
  return `#${lr.toString(16).padStart(2, "0")}${lg.toString(16).padStart(2, "0")}${lb.toString(16).padStart(2, "0")}`;
}
