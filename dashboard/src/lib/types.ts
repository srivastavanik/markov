// Types matching the Python server WebSocket schema

export interface AgentState {
  id: string;
  name: string;
  family: string;
  color: string;
  provider: string;
  model: string;
  tier: number; // 1=Boss, 2=Lieutenant, 3=Soldier
  temperature: number;
  position: [number, number];
  alive: boolean;
  eliminated_by: string | null;
  eliminated_round: number | null;
}

export interface FamilyConfig {
  name: string;
  provider: string;
  color: string;
  agent_ids: string[];
}

export interface GameEvent {
  type:
    | "move"
    | "stay"
    | "elimination"
    | "mutual_elimination"
    | "failed_move"
    | "failed_eliminate";
  agent_id: string;
  details: Record<string, unknown>;
}

export interface MessageData {
  round: number;
  sender: string;
  sender_name: string;
  channel: "family" | "dm" | "broadcast";
  recipient: string | null;
  content: string;
  family: string;
}

export interface HighlightData {
  round: number;
  agent_id: string;
  type: string;
  severity: "critical" | "high" | "medium";
  description: string;
  excerpt: string;
}

export interface MaliceIndicators {
  elimination_planning: boolean;
  targets_mentioned: string[];
  deception_planning: boolean;
  manipulation_planning: boolean;
  betrayal_planning: boolean;
  exploitation_language: boolean;
  unprompted: boolean;
  sophistication: number;
  raw_excerpts: string[];
}

export interface BetrayalData {
  detected: boolean;
  target_family_member: string | null;
  nature: string | null;
  rationalization: string | null;
  guilt_expressed: boolean;
}

export interface SafetyArtifacts {
  hedging: boolean;
  ethical_reasoning: boolean;
  euphemisms_used: string[];
  refusal_language: boolean;
  rationalization_type: string | null;
}

export interface AgentAnalysis {
  deception_delta: number;
  malice: MaliceIndicators;
  family_sentiment: Record<string, number>;
  betrayal: BetrayalData;
  safety_artifacts: SafetyArtifacts;
}

export interface RoundMessages {
  family_discussions: Array<{
    family: string;
    transcript: Array<{
      agent: string;
      agent_id: string;
      tier: number;
      discussion_round: number;
      content: string;
    }>;
  }>;
  direct_messages: MessageData[];
  broadcasts: MessageData[];
  family_messages: MessageData[];
}

export interface RoundData {
  game_id?: string;
  round: number;
  grid: {
    size: number;
    agents: AgentState[];
  };
  events: GameEvent[];
  thoughts: Record<string, string>;
  messages: RoundMessages;
  analysis: Record<string, AgentAnalysis>;
  highlights: HighlightData[];
  alive_count: number;
  game_over: boolean;
  winner: string | null;
}

export interface GameInitData {
  type: "game_init";
  game_id?: string;
  grid_size: number;
  agents: Record<string, Omit<AgentState, "color" | "eliminated_by" | "eliminated_round">>;
  families: FamilyConfig[];
  total_rounds?: number;
  result?: {
    winner_name: string | null;
    total_rounds: number;
    final_reflection: string | null;
  };
}

export interface GameOverData {
  type: "game_over";
  game_id?: string;
  winner: string | null;
  winner_family: string | null;
  total_rounds: number;
  final_reflection: string | null;
  cancelled?: boolean;
}
