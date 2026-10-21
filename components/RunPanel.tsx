import React from 'react';
import { Workflow } from 'lucide-react';
import { AgentRun } from '../types';
import { FINISH_LABELS, TraceStepList, totalSeconds } from './AgentTrace';

interface RunPanelProps {
  run: AgentRun | null;
  isRunning: boolean;
}

// The latest request's path through the agent graph, from the backend trace.
export const RunPanel: React.FC<RunPanelProps> = ({ run, isRunning }) => {
  if (!run) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-400 bg-slate-50/50 p-8 text-center">
        <Workflow size={56} className="mb-4 text-slate-300" />
        <p className="font-medium">{isRunning ? 'Running…' : 'No run yet'}</p>
        <p className="text-sm max-w-xs">
          Each request's route through the supervisor and the specialists appears here.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="p-6 border-b border-slate-100">
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <Workflow size={20} /> Latest agent run
        </h2>
        <div className="text-sm text-slate-500 mt-1 flex flex-wrap gap-x-3">
          <span>{run.steps} specialist step{run.steps === 1 ? '' : 's'}</span>
          <span>{FINISH_LABELS[run.finishReason]}</span>
          <span>{totalSeconds(run)} s</span>
          {isRunning && <span className="text-brand-600">new request running…</span>}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-6 text-sm">
        <TraceStepList trace={run.trace} />
        <p className="mt-6 text-xs text-slate-400">request {run.requestId}</p>
      </div>
    </div>
  );
};
