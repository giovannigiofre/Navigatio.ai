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

export interface AgentAction {
  tool: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
}

export interface Message {
  id: string;
  sender: Sender;
  content: string;
  timestamp: Date;
  agentActions?: AgentAction[]; // Trace what the agent is doing (e.g. "Checking flights...")
}

export interface ChatState {
  messages: Message[];
  isTyping: boolean;
}