export enum Sender {
  User = 'USER',
  AI = 'AI',
  System = 'SYSTEM'
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
  failed?: boolean; // A user message whose request failed: not sent as history
}
