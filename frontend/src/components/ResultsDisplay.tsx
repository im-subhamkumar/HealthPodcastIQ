import React, { useContext } from 'react';
import { SummaryResult, SummaryLength, VoiceOption } from '../types';
import QACard from './QACard';
import { useTextToSpeech } from '../hooks/useTextToSpeech';
import { RefreshIcon, SpeakerIcon, MaleIcon, FemaleIcon } from './icons';
import { SettingsContext } from '../contexts/SettingsContext';

interface ResultsDisplayProps {
  result: SummaryResult;
  onReset: () => void;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result, onReset }) => {
  const { length, setLength, voice, setVoice } = useContext(SettingsContext);
  const { speak, isSpeaking } = useTextToSpeech();

  if (!length || !setLength || !voice || !setVoice) {
    return null; // Or a loading state, as context might not be available instantly
  }

  const handlePlayOverallSummary = () => {
    speak(result.overallSummary[length]);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {result.thumbnailUrl && (
        <div className="mb-6 rounded-xl overflow-hidden shadow-lg border border-slate-200">
          <img
            src={result.thumbnailUrl}
            alt={`${result.title} thumbnail`}
            className="w-full h-auto object-cover"
          />
        </div>
      )}

      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <h2 className="text-3xl font-bold text-slate-800 truncate flex items-center gap-3" title={result.title}>
          {result.title}
          {result.cached && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
              <svg className="-ml-0.5 mr-1.5 h-2 w-2 text-emerald-400" fill="currentColor" viewBox="0 0 8 8">
                <circle cx="4" cy="4" r="3" />
              </svg>
              Cached
            </span>
          )}
        </h2>
        <button
          onClick={onReset}
          className="flex items-center justify-center gap-2 bg-slate-100 text-slate-600 font-semibold py-2 px-4 rounded-lg hover:bg-slate-200 transition-colors text-sm"
        >
          <RefreshIcon className="w-4 h-4" />
          Process Another
        </button>
      </div>

      {/* UI Controls Panel - Updated with purple/indigo theme */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col sm:flex-row justify-center items-center gap-4 sm:gap-6">
        {/* Summary Length Controls */}
        <div className="p-1 bg-slate-200 rounded-lg flex space-x-1">
          {(Object.values(SummaryLength)).map((len) => (
            <button
              key={len}
              onClick={() => setLength(len)}
              className={`px-4 py-1.5 rounded-md text-sm font-semibold transition-all duration-200 ${length === len
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'text-slate-600 hover:bg-slate-300'
                }`}
            >
              {len.charAt(0).toUpperCase() + len.slice(1)}
            </button>
          ))}
        </div>
        {/* Voice Selection Controls */}
        <div className="p-1 bg-slate-200 rounded-lg flex space-x-1">
          {(Object.values(VoiceOption)).map((v) => (
            <button
              key={v}
              onClick={() => setVoice(v)}
              className={`px-4 py-1.5 rounded-md text-sm font-semibold transition-all duration-200 flex items-center gap-2 ${voice === v
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'text-slate-600 hover:bg-slate-300'
                }`}
            >
              {v === VoiceOption.FEMALE ? <FemaleIcon className="w-4 h-4" /> : <MaleIcon className="w-4 h-4" />}
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>


      {/* Overall Summary - Enhanced styling */}
      <div className="p-6 bg-purple-50/50 border border-purple-200 rounded-xl shadow-sm">
        <div className="flex justify-between items-start mb-3">
          <h3 className="text-xl font-bold text-purple-800">Overall Summary</h3>
          <button
            onClick={handlePlayOverallSummary}
            disabled={isSpeaking}
            className="p-2 text-purple-600 hover:bg-purple-100 rounded-full transition disabled:opacity-50"
            aria-label="Play overall summary"
          >
            <SpeakerIcon className="w-5 h-5" />
          </button>
        </div>
        <p className="text-slate-700 leading-relaxed">{result.overallSummary[length]}</p>
      </div>

      {/* Q&A Section */}
      <div>
        <h3 className="text-2xl font-bold mb-4 text-slate-800">Questions & Answers</h3>
        {result.qaPairs && result.qaPairs.length > 0 ? (
          <div className="space-y-4">
            {result.qaPairs.map((qaPair) => (
              <QACard key={qaPair.id} qaPair={qaPair} />
            ))}
          </div>
        ) : (
          <div className="p-6 bg-amber-50 border border-amber-200 rounded-xl">
            <p className="text-amber-800">
              No Q&A pairs were generated. This may occur if:
            </p>
            <ul className="list-disc list-inside mt-2 text-amber-700 text-sm space-y-1">
              <li>The content is not health/fitness/nutrition related</li>
              <li>The transcript is too short</li>
              <li>The models need more training data</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsDisplay;