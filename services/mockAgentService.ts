import { Message, Sender, TripPlan, ActivityType, AgentAction } from '../types';

/**
 * This service mocks the behavior of the future Backend Agent.
 * In production, this would be replaced by API calls to your LangChain/Gemini backend.
 */

const SAMPLE_TRIP: TripPlan = {
  destination: "Tokyo, Japan",
  startDate: "2024-04-10",
  endDate: "2024-04-14",
  totalBudget: 3500,
  currency: "USD",
  itinerary: [
    {
      day: 1,
      date: "2024-04-10",
      activities: [
        {
          id: 'a1',
          time: '14:00',
          title: 'Arrival at Narita',
          description: 'Land at NRT, take Narita Express to Shinjuku.',
          type: ActivityType.Flight,
          cost: 35
        },
        {
          id: 'a2',
          time: '16:30',
          title: 'Check-in: Hotel Gracery Shinjuku',
          description: 'The famous Godzilla hotel.',
          type: ActivityType.Hotel,
          cost: 200
        },
        {
          id: 'a3',
          time: '19:00',
          title: 'Dinner at Omoide Yokocho',
          description: 'Experience authentic Yakitori in "Piss Alley".',
          type: ActivityType.Food,
          cost: 40
        }
      ]
    },
    {
      day: 2,
      date: "2024-04-11",
      activities: [
        {
          id: 'a4',
          time: '09:00',
          title: 'Meiji Shrine',
          description: 'Morning walk through the forest to the shrine.',
          type: ActivityType.Sightseeing,
          location: 'Shibuya',
          cost: 0
        },
        {
          id: 'a5',
          time: '13:00',
          title: 'Shopping in Harajuku',
          description: 'Explore Takeshita Street.',
          type: ActivityType.Leisure,
          cost: 100
        }
      ]
    }
  ]
};

export const simulateAgentResponse = async (
  userMessage: string,
  currentPlan: TripPlan | null,
  onProgress: (action: AgentAction) => void
): Promise<{ text: string; updatedPlan: TripPlan | null }> => {

  // Simulate network delay and agent "thinking"
  await new Promise(resolve => setTimeout(resolve, 800));

  // Simulate Tool Use 1: Search
  onProgress({
    tool: 'SearchTool',
    status: 'running',
    message: `Searching for attractions in ${currentPlan?.destination || 'destination'}...`
  });
  await new Promise(resolve => setTimeout(resolve, 1500));
  onProgress({
    tool: 'SearchTool',
    status: 'completed',
    message: 'Found 12 relevant locations.'
  });

  // Simulate Tool Use 2: Price Check
  onProgress({
    tool: 'PriceCheckAPI',
    status: 'running',
    message: 'Verifying opening hours and costs...'
  });
  await new Promise(resolve => setTimeout(resolve, 1200));
  onProgress({
    tool: 'PriceCheckAPI',
    status: 'completed',
    message: 'Updated pricing data retrieved.'
  });

  // Determine response based on if it's the first interaction or an update
  if (!currentPlan) {
    return {
      text: "I've put together an initial 2-day itinerary for Tokyo based on your preferences. We start in Shinjuku and move to Shibuya. How does this look?",
      updatedPlan: SAMPLE_TRIP
    };
  } else {
    // Simulate adding an item to the existing plan
    const newPlan = JSON.parse(JSON.stringify(currentPlan));
    if (newPlan.itinerary[1]) {
      newPlan.itinerary[1].activities.push({
        id: 'a6',
        time: '18:00',
        title: 'Shibuya Sky Sunset',
        description: 'Added per your request. Best view of the crossing.',
        type: ActivityType.Sightseeing,
        cost: 25
      });
    }
    
    return {
      text: "I've added Shibuya Sky to Day 2 around sunset. The tickets are about $25. Would you like me to look for dinner options nearby?",
      updatedPlan: newPlan
    };
  }
};