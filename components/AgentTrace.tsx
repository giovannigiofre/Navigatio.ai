import React from 'react';
import { AlertCircle, CheckCircle2, Cpu } from 'lucide-react';
import { AgentRun, TraceStep } from '../types';

export const FINISH_LABELS: Record<AgentRun['finishReason'], string> = {
  supervisor: 'finished by the supervisor',
  step_budget: 'stopped: step budget reached',
  router_error: 'stopped: invalid routing output',
};

export const totalSeconds = (run: AgentRun): string =>
  (run.trace.reduce((sum, step) => sum + step.latency_ms, 0) / 1000).toFixed(1);

const hasToolError = (step: TraceStep) =>
  step.tool_calls?.some(call => call.status === 'error') ?? false;

const describe = (step: TraceStep): string => {
  if (step.decision) return `${step.node} → ${step.decision}`;
  const calls = step.tool_calls ?? [];
  if (calls.length === 0) return step.node;
  const outcomes = calls.map(call =>
    call.status === 'ok' ? `${call.tool}: ok` : `${call.tool}: ${call.error_kind ?? 'error'}`
  );
  return `${step.node} (${outcomes.join(', ')})`;
};

// The nodes that ran, in order, with supervisor decisions, tool outcomes and latency.
export const TraceStepList: React.FC<{ trace: TraceStep[] }> = ({ trace }) => (
  <ol className="space-y-1.5">
    {trace.map((step, index) => (
      <li key={index} className="flex items-start gap-2">
        {hasToolError(step) ? (
          <AlertCircle size={12} className="mt-0.5 shrink-0 text-red-500" />
        ) : (
          <CheckCircle2 size={12} className="mt-0.5 shrink-0 text-emerald-500" />
        )}
        <div className="min-w-0">
          <div className="text-slate-700">
            {describe(step)}
            <span className="ml-2 text-slate-400">{Math.round(step.latency_ms)} ms</span>
          </div>
          {step.reason && <div className="text-slate-400">{step.reason}</div>}
        </div>
      </li>
    ))}
  </ol>
);

// Collapsible summary of one run, shown with each reply in the chat.
export const AgentTrace: React.FC<{ run: AgentRun }> = ({ run }) => {
  const plural = run.steps === 1 ? '' : 's';

  return (
    <details className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-xs text-slate-600 mb-1">
      <summary className="flex items-center gap-2 cursor-pointer font-medium text-slate-500 text-[11px]">
        <Cpu size={12} />
        Agent trace · {run.steps} specialist step{plural} · {FINISH_LABELS[run.finishReason]} ·{' '}
        {totalSeconds(run)} s
      </summary>
      <div className="mt-2">
        <TraceStepList trace={run.trace} />
      </div>
      <div className="mt-2 text-[10px] text-slate-400">request {run.requestId}</div>
    </details>
  );
};
