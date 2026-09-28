export enum Sender {
  User = 'USER',
  AI = 'AI',
  System = 'SYSTEM'
}

export enum ActivityType {
  Flight = 'FLIGHT',
  Hotel = 'HOTEL',
  Food = 'FOOD',
  Sightseeing = 'SIGHTSEEING',
  Transport = 'TRANSPORT',
  Leisure = 'LEISURE'
}

export interface Activity {
  id: string;
  time: string;
  title: string;
  description: string;
  type: ActivityType;
  cost?: number;
  location?: string;
  duration?: string;
}

export interface DayPlan {
  day: number;
  date: string;
  activities: Activity[];
}

export interface TripPlan {
  destination: string;
  startDate: string;
  endDate: string;
  totalBudget: number;
  currency: string;
  itinerary: DayPlan[];
}

// One turn of conversation history, as sent to the backend.
export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ToolCallTrace {
  tool: string;
  status: 'ok' | 'error';
  error_kind?: string;
  days?: number;
}

// One node execution in the backend graph, as returned in the /plan trace.
export interface TraceStep {
  node: string;
  latency_ms: number;
  decision?: string;
  reason?: string;
  tool_calls?: ToolCallTrace[];
}

export type FinishReason = 'supervisor' | 'step_budget' | 'router_error';

export interface AgentRun {
  requestId: string;
  steps: number;
  finishReason: FinishReason;
  trace: TraceStep[];
}

export interface Message {
  id: string;
  sender: Sender;
  content: string;
  timestamp: Date;
  run?: AgentRun; // How the backend produced this reply
}

export interface ChatState {
  messages: Message[];
  isTyping: boolean;
}