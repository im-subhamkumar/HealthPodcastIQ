import React from 'react';
import { EpisodeSequenceResult } from '../types';
import { RefreshIcon } from './icons';
import SequenceSegmentCard from './SequenceSegmentCard';

interface EpisodeSequenceDisplayProps {
  result: EpisodeSequenceResult;
  onReset: () => void;
}

const EpisodeSequenceDisplay: React.FC<EpisodeSequenceDisplayProps> = ({ result, onReset }) => {
  return (
    <div className="max-w-4xl mx-auto space-y-10 animate-fade-in py-4">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-slate-100 pb-6">
        <h2 className="text-3xl font-extrabold text-slate-900 leading-tight" title={result.sequenceTitle}>
          {result.sequenceTitle}
        </h2>
        <button
          onClick={onReset}
          className="flex items-center gap-2 bg-slate-50 text-slate-600 font-bold py-3 px-5 rounded-xl border border-slate-200 hover:bg-white hover:border-slate-300 hover:shadow-sm transition-all text-xs uppercase tracking-wider"
        >
          <RefreshIcon className="w-5 h-5 text-slate-400" />
          Create Another
        </button>
      </div>

      {/* Introduction Block */}
      <div className="p-8 bg-indigo-50/40 border border-indigo-100 rounded-3xl shadow-sm relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-100/30 rounded-full -mr-16 -mt-16 transition-transform group-hover:scale-110 duration-700"></div>
        <h3 className="text-lg font-black mb-4 text-indigo-900 flex items-center gap-2 uppercase tracking-tight">
          <span className="w-2 h-6 bg-indigo-500 rounded-full"></span>
          Sequence Introduction
        </h3>
        <p className="text-slate-700 leading-relaxed text-lg font-medium relative z-10">
          {result.sequenceIntroduction}
        </p>
      </div>

      {/* Segments Section */}
      <div className="space-y-8">
        <h3 className="text-2xl font-black text-slate-900 flex items-center gap-3">
          Your Episode Sequence
        </h3>
        <div className="relative">
          {/* Main timeline line is now handled inside cards for better control, 
              but we can keep a subtle base line if needed */}
          <div className="space-y-4">
            {result.segments.map((segment, index) => (
              <SequenceSegmentCard
                key={segment.id}
                segment={segment}
                index={index}
                isLast={index === result.segments.length - 1}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EpisodeSequenceDisplay;
