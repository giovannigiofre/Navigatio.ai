import React from 'react';
import { Message, Sender } from '../types';
import { Bot, User } from 'lucide-react';
import { AgentTrace } from './AgentTrace';

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
          
          {message.run && <AgentTrace run={message.run} />}

          {/* Message Text */}
          <div className={`
            p-4 rounded-2xl shadow-sm text-sm leading-relaxed whitespace-pre-wrap break-words
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