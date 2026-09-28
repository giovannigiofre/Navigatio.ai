import React, { useState, useRef, useEffect } from 'react';
import { SendHorizontal, Sparkles, Command } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { VisualizationPanel } from './components/VisualizationPanel';
import { ChatTurn, Message, Sender, TripPlan } from './types';
import { sendMessageToAgent } from './services/api';

// Initial greeting message (UI only: not sent to the backend as history)
const INITIAL_MESSAGE: Message = {
  id: 'init-1',
  sender: Sender.AI,
  content: "Hello! Where and when would you like to travel? I can check the live weather forecast for trips in the next 16 days, suggest things to do, and sketch illustrative flight and hotel options.",
  timestamp: new Date()
};

// Earlier user/assistant turns, oldest first, for multi-turn context on the backend.
const toHistory = (messages: Message[]): ChatTurn[] =>
  messages
    .filter(m => m.id !== INITIAL_MESSAGE.id && m.sender !== Sender.System)
    .map(m => ({ role: m.sender === Sender.User ? 'user' : 'assistant', content: m.content }));

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState('');
  // The backend does not return a structured itinerary yet, so this stays empty.
  const [currentPlan] = useState<TripPlan | null>(null);
  const [isTyping, setIsTyping] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSendMessage = async () => {
    if (!input.trim() || isTyping) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      sender: Sender.User,
      content: input,
      timestamp: new Date()
    };

    const history = toHistory(messages);
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const reply = await sendMessageToAgent(userMsg.content, history);

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: Sender.AI,
        content: reply.text,
        timestamp: new Date(),
        run: reply.run
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error("Agent Error:", error);
      const reason = error instanceof Error ? error.message : 'Unknown error.';
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: Sender.System,
        content: `Sorry, the request failed: ${reason}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-slate-50 font-sans text-slate-900">

      {/* LEFT: Chat Interface */}
      <div className="w-full md:w-[450px] lg:w-[500px] flex flex-col border-r border-slate-200 bg-white shadow-xl z-20">

        {/* Header */}
        <div className="h-16 flex items-center px-6 border-b border-slate-100 bg-white/80 backdrop-blur z-10">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-indigo-600 flex items-center justify-center text-white mr-3 shadow-lg shadow-brand-500/20">
            <Sparkles size={18} />
          </div>
          <div>
            <h1 className="font-bold text-slate-800">Navigatio.ai</h1>
            <p className="text-xs text-slate-500 font-medium">Agentic Trip Planner</p>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50/50 scroll-smooth">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* Pending indicator: the backend answers once the whole run is done (no streaming) */}
          {isTyping && (
            <div className="flex w-full mb-6 justify-start">
              <div className="flex flex-col gap-2 max-w-[85%] ml-11">
                <div className="bg-white border border-brand-100 rounded-lg p-3 text-xs text-slate-600 shadow-sm animate-pulse flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-bounce"></div>
                  <span>The agents are working on it. This can take up to a minute.</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-white border-t border-slate-100">
          <div className="relative flex items-end gap-2 bg-slate-50 border border-slate-200 rounded-2xl p-2 shadow-inner focus-within:ring-2 focus-within:ring-brand-100 focus-within:border-brand-400 transition-all">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your ideal trip..."
              className="w-full bg-transparent border-none focus:ring-0 resize-none max-h-32 min-h-[44px] py-2.5 px-3 text-sm text-slate-800 placeholder:text-slate-400"
              rows={1}
            />
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || isTyping}
              className={`
                p-2.5 rounded-xl mb-0.5 transition-all duration-200
                ${input.trim() && !isTyping
                  ? 'bg-brand-600 text-white shadow-md shadow-brand-500/20 hover:bg-brand-700 hover:scale-105 active:scale-95'
                  : 'bg-slate-200 text-slate-400 cursor-not-allowed'}
              `}
            >
              <SendHorizontal size={20} />
            </button>
          </div>
          <div className="mt-2 text-center">
            <p className="text-[10px] text-slate-400 flex items-center justify-center gap-1">
              <Command size={10} /> Weather: live Open-Meteo forecast. Flights and hotels: illustrative, not live prices.
            </p>
          </div>
        </div>
      </div>

      {/* RIGHT: Visualization Panel */}
      <div className="hidden md:flex flex-1 flex-col bg-slate-100 overflow-hidden relative">
        <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none"
          style={{ backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
        </div>
        <div className="z-10 h-full p-4">
          <div className="h-full rounded-2xl overflow-hidden shadow-xl border border-slate-200/60 bg-white">
            <VisualizationPanel plan={currentPlan} isLoading={isTyping && !currentPlan} />
          </div>
        </div>
      </div>

    </div>
  );
};

export default App;