import React from 'react';
import { TripPlan, Activity, ActivityType } from '../types';
import { Plane, Hotel, Utensils, Camera, Bus, MapPin, DollarSign, Clock } from 'lucide-react';

interface TripTimelineProps {
  plan: TripPlan;
}

const getActivityIcon = (type: ActivityType) => {
  switch (type) {
    case ActivityType.Flight: return <Plane size={16} />;
    case ActivityType.Hotel: return <Hotel size={16} />;
    case ActivityType.Food: return <Utensils size={16} />;
    case ActivityType.Sightseeing: return <Camera size={16} />;
    case ActivityType.Transport: return <Bus size={16} />;
    default: return <MapPin size={16} />;
  }
};

const getActivityColor = (type: ActivityType) => {
  switch (type) {
    case ActivityType.Flight: return 'bg-blue-100 text-blue-600 border-blue-200';
    case ActivityType.Hotel: return 'bg-indigo-100 text-indigo-600 border-indigo-200';
    case ActivityType.Food: return 'bg-orange-100 text-orange-600 border-orange-200';
    case ActivityType.Sightseeing: return 'bg-emerald-100 text-emerald-600 border-emerald-200';
    default: return 'bg-slate-100 text-slate-600 border-slate-200';
  }
};

export const TripTimeline: React.FC<TripTimelineProps> = ({ plan }) => {
  return (
    <div className="space-y-8 p-1">
      {plan.itinerary.map((day) => (
        <div key={day.day} className="relative">
          {/* Day Header */}
          <div className="sticky top-0 z-10 bg-white/95 backdrop-blur py-3 border-b border-slate-100 mb-4 flex justify-between items-center">
            <h3 className="font-bold text-lg text-slate-800">Day {day.day}</h3>
            <span className="text-sm font-medium text-slate-500 bg-slate-100 px-2 py-1 rounded">
              {day.date}
            </span>
          </div>

          <div className="relative pl-6 border-l-2 border-slate-100 space-y-6 ml-3">
            {day.activities.map((activity) => (
              <div key={activity.id} className="relative group cursor-pointer transition-transform hover:-translate-y-1">
                {/* Timeline Dot */}
                <div className={`
                  absolute -left-[31px] top-0 w-8 h-8 rounded-full flex items-center justify-center border-2 bg-white
                  ${getActivityColor(activity.type)}
                `}>
                  {getActivityIcon(activity.type)}
                </div>

                {/* Card */}
                <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-slate-900">{activity.title}</h4>
                    <span className="flex items-center text-xs font-medium text-slate-500 bg-slate-50 px-2 py-1 rounded-full border border-slate-100">
                      <Clock size={12} className="mr-1" /> {activity.time}
                    </span>
                  </div>
                  
                  <p className="text-sm text-slate-600 mb-3 line-clamp-2">
                    {activity.description}
                  </p>

                  <div className="flex items-center gap-4 text-xs text-slate-500">
                    {activity.location && (
                      <span className="flex items-center gap-1">
                        <MapPin size={12} /> {activity.location}
                      </span>
                    )}
                    {activity.cost !== undefined && (
                      <span className="flex items-center gap-1 font-medium text-slate-700">
                        <DollarSign size={12} /> {plan.currency} {activity.cost}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};