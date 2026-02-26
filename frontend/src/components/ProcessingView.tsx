import React, { useState, useEffect } from 'react';

const processingSteps = [
  "Extracting audio from source...",
  "Transcribing speech to text...",
  "Identifying speakers and questions...",
  "Generating multi-length summaries...",
  "Finalizing results...",
];

const ProcessingView: React.FC = () => {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime((prevTime) => prevTime + 1);
    }, 1000);

    const stepInterval = setInterval(() => {
      setCurrentStep((prevStep) => (prevStep + 1) % processingSteps.length);
    }, 2000);

    return () => {
      clearInterval(timer);
      clearInterval(stepInterval);
    };
  }, []);
  
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Estimated time for a long podcast
  const totalTimeEstimate = 5 * 60; // 5 minutes
  const progress = Math.min((elapsedTime / totalTimeEstimate) * 100, 100);

  return (
    <div className="text-center py-8 px-4 flex flex-col items-center">
      {/* Large circular progress indicator matching image design */}
      <div className="relative w-24 h-24 mb-6">
          {/* Outer grey circle */}
          <div className="absolute inset-0 border-4 border-slate-200 rounded-full"></div>
          {/* Progress arc using SVG for better control */}
          <svg className="absolute inset-0 w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="46"
              fill="none"
              stroke="rgb(99, 102, 241)"
              strokeWidth="8"
              strokeDasharray={`${2 * Math.PI * 46}`}
              strokeDashoffset={`${2 * Math.PI * 46 * (1 - progress / 100)}`}
              strokeLinecap="round"
              className="transition-all duration-500"
            />
          </svg>
          {/* Percentage display inside circle */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xl font-bold text-indigo-600">{Math.floor(progress)}%</span>
          </div>
      </div>
      
      <h2 className="text-3xl font-bold text-slate-800 mb-2">Processing Podcast</h2>
      <p className="text-slate-500 text-sm mb-8">
        This may take a few minutes for long videos. Please don't close this window.
      </p>

      {/* Dark status box matching image design */}
      <div className="w-full max-w-sm bg-slate-800 rounded-lg p-5 shadow-lg">
        <div className="flex justify-between items-center font-mono text-sm mb-3">
            <span className="text-slate-400">Time Elapsed:</span>
            <span className="text-white font-semibold tracking-wider">{formatTime(elapsedTime)}</span>
        </div>
        {/* Progress bar */}
        <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden mb-4">
            <div 
              className="bg-indigo-500 h-full rounded-full transition-all duration-500 ease-in-out" 
              style={{ width: `${progress}%` }}
            ></div>
        </div>
        {/* Current step indicator */}
        <p className="text-center text-sm text-indigo-200 font-medium h-5 animate-pulse">
            {processingSteps[currentStep]}
        </p>
      </div>
    </div>
  );
};

export default ProcessingView;