import React from 'react';
import { EpisodeSegment } from '../types';

interface SequenceSegmentCardProps {
  segment: EpisodeSegment;
  index: number;
  isLast: boolean;
}

const SequenceSegmentCard: React.FC<SequenceSegmentCardProps> = ({ segment, index, isLast }) => {
  return (
    <div className={`relative ${isLast ? 'pb-2' : 'pb-10'}`}>
      {/* Timeline Line */}
      {!isLast && (
        <div className="absolute left-[15px] top-[32px] w-0.5 h-full bg-slate-100"></div>
      )}

      {/* Green numbered circle */}
      <div className="absolute left-0 top-1 w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center shadow-sm z-10 border-2 border-white">
        <span className="text-sm font-bold text-white leading-none">{index + 1}</span>
      </div>

      <div className="ml-12 p-6 bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md transition-all duration-300 group">
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {/* Source Badge */}
          <span className="inline-flex items-center px-3 py-1 rounded-full text-[11px] font-bold tracking-wide uppercase bg-sky-50 text-sky-700 border border-sky-100">
            Source: Podcast {segment.sourcePodcast}
          </span>
          {/* Concept Badge */}
          <span className="inline-flex items-center px-3 py-1 rounded-full text-[11px] font-bold tracking-wide uppercase bg-indigo-50 text-indigo-700 border border-indigo-100">
            Concept: {segment.keyConcept}
          </span>
        </div>

        <h4 className="font-extrabold text-xl text-slate-800 mb-2 group-hover:text-emerald-700 transition-colors">
          {segment.title}
        </h4>
        <p className="text-slate-600 leading-relaxed text-[15px]">
          {segment.summary}
        </p>
      </div>
    </div>
  );
};

export default SequenceSegmentCard;
