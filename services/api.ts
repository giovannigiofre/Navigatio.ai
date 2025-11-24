import { Message, TripPlan, AgentAction } from '../types';

const API_URL = 'http://localhost:8000';

interface PlanResponse {
    response: string;
    trace: string[];
}

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
            throw new Error(`API Error: ${res.statusText}`);
        }

        const data: PlanResponse = await res.json();

        // 2. Simulate "Streaming" the trace for UI effect
        // In a real app, we would use Server Sent Events (SSE) or WebSockets
        for (const traceMsg of data.trace) {
            // Simple heuristic to guess the tool/agent from the message content
            let tool = 'Agent';
            if (traceMsg.includes('Flight')) tool = 'FlightExpert';
            else if (traceMsg.includes('Hotel')) tool = 'HotelScout';
            else if (traceMsg.includes('Activity')) tool = 'ActivityCurator';
            else if (traceMsg.includes('Supervisor')) tool = 'Supervisor';

            onProgress({
                tool,
                status: 'completed',
                message: traceMsg.substring(0, 50) + '...' // Truncate for UI
            });

            // Artificial delay to let the user see the steps
            await new Promise(r => setTimeout(r, 500));
        }

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
