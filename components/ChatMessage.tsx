import React from 'react';
import { Message, Sender, AgentAction } from '../types';
import { Bot, User, Cpu, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.sender === Sender.User;

  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'} gap-3`}>
        {/* Avatar */}
        <div className={`
          flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
          ${isUser ? 'bg-brand-600 text-white' : 'bg-emerald-600 text-white'}
        `}>
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>

        {/* Content Wrapper */}
        <div className="flex flex-col gap-2">
          
          {/* Agent Actions (Thinking Process) */}
          {message.agentActions && message.agentActions.length > 0 && (
            <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-xs text-slate-600 mb-1 space-y-2">
              <div className="flex items-center gap-2 font-medium text-slate-500 uppercase tracking-wider text-[10px]">
                <Cpu size={12} /> Agent Thought Process
              </div>
              {message.agentActions.map((action, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  {action.status === 'running' && <Loader2 size={12} className="animate-spin text-brand-500" />}
                  {action.status === 'completed' && <CheckCircle2 size={12} className="text-emerald-500" />}
                  {action.status === 'failed' && <AlertCircle size={12} className="text-red-500" />}
                  <span className={action.status === 'running' ? 'text-slate-800' : 'text-slate-500'}>
                    {action.message}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Message Text */}
          <div className={`
            p-4 rounded-2xl shadow-sm text-sm leading-relaxed
            ${isUser 
              ? 'bg-brand-600 text-white rounded-tr-none' 
              : 'bg-white border border-slate-100 text-slate-800 rounded-tl-none'}
          `}>
             {message.content}
          </div>
          
          <div className={`text-[10px] text-slate-400 ${isUser ? 'text-right' : 'text-left'}`}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    </div>
  );
};