import { AgentRun, ChatTurn, FinishReason, TraceStep } from '../types';

const API_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// The backend bounds a run at 90 s by default; give up a little later than that.
const REQUEST_TIMEOUT_MS = 120_000;

interface PlanResponse {
  request_id: string;
  response: string;
  steps: number;
  finish_reason: FinishReason;
  trace: TraceStep[];
}

interface ErrorResponse {
  error: { code: string; message: string; request_id?: string };
}

export interface AgentReply {
  text: string;
  run: AgentRun;
}

export const sendMessageToAgent = async (
  message: string,
  history: ChatTurn[]
): Promise<AgentReply> => {
  const res = await fetch(`${API_URL}/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });

  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as ErrorResponse | null;
    throw new Error(body?.error?.message ?? `Request failed (HTTP ${res.status}).`);
  }

  const data = (await res.json()) as PlanResponse;
  return {
    text: data.response,
    run: {
      requestId: data.request_id,
      steps: data.steps,
      finishReason: data.finish_reason,
      trace: data.trace,
    },
  };
};
