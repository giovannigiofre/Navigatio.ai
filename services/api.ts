import { Message, TripPlan, AgentAction } from '../types';

const API_URL = 'http://localhost:8000';

interface ToolCallTrace {
    tool: string;
    status: 'ok' | 'error';
    error_kind?: string;
    days?: number;
}

interface TraceStep {
    node: string;
    latency_ms: number;
    decision?: string;
    reason?: string;
    tool_calls?: ToolCallTrace[];
}

interface PlanResponse {
    request_id: string;
    response: string;
    steps: number;
    finish_reason: 'supervisor' | 'step_budget' | 'router_error';
    trace: TraceStep[];
}

interface ErrorResponse {
    error: { code: string; message: string; request_id?: string };
}

const describeStep = (step: TraceStep): string => {
    const ms = `${Math.round(step.latency_ms)} ms`;
    if (step.decision) return `${step.node} → ${step.decision} (${ms})`;
    const failed = step.tool_calls?.filter(call => call.status === 'error') ?? [];
    if (failed.length) return `${step.node}: ${failed.map(call => call.error_kind).join(', ')} (${ms})`;
    return `${step.node} (${ms})`;
};

export const sendMessageToAgent = async (
    message: string,
    currentPlan: TripPlan | null,
    onProgress: (action: AgentAction) => void
): Promise<{ text: string; updatedPlan: TripPlan | null }> => {

    try {
        // 1. Send request to backend
        onProgress({
            tool: 'Orchestrator',
            status: 'running',
            message: 'Connecting to Agent Team...'
        });

        const res = await fetch(`${API_URL}/plan`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                history: [] // We could pass history here if the backend supported it fully
            }),
        });

        if (!res.ok) {
            const body = (await res.json().catch(() => null)) as ErrorResponse | null;
            throw new Error(body?.error?.message ?? `API Error: ${res.status}`);
        }

        const data: PlanResponse = await res.json();

        // 2. Show the executed steps, as reported by the backend trace
        data.trace.forEach((step, index) => {
            onProgress({
                tool: `${index}:${step.node}`,
                status: step.tool_calls?.some(call => call.status === 'error') ? 'failed' : 'completed',
                message: describeStep(step)
            });
        });

        // 3. Return final response
        return {
            text: data.response,
            updatedPlan: currentPlan // For now, we don't parse the plan from the text yet
        };

    } catch (error) {
        console.error("Agent API Error:", error);
        onProgress({
            tool: 'System',
            status: 'failed',
            message: 'Connection failed.'
        });
        throw error;
    }
};
