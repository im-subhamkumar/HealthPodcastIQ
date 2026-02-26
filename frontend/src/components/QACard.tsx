import React, { useContext } from 'react';
import { QAPair } from '../types';
import { useTextToSpeech } from '../hooks/useTextToSpeech';
import { PlayIcon, StopIcon } from './icons';
import { SettingsContext } from '../contexts/SettingsContext';
import FactCheckDisplay from './FactCheckDisplay';

interface QACardProps {
  qaPair: QAPair;
}

const QACard: React.FC<QACardProps> = ({ qaPair }) => {
  const { length } = useContext(SettingsContext);
  const { speak, stop, isSpeaking, currentlySpeakingText } = useTextToSpeech();
  
  if(!length) return null;

  const answerText = qaPair.answers[length];
  const questionText = qaPair.question;
  const isAnswerSpeaking = isSpeaking && currentlySpeakingText === answerText;
  const isQuestionSpeaking = isSpeaking && currentlySpeakingText === questionText;

  const handleTogglePlayQuestion = () => {
    if (isQuestionSpeaking) {
      stop();
    } else {
      speak(questionText);
    }
  };

  const handleTogglePlayAnswer = () => {
    if (isAnswerSpeaking) {
      stop();
    } else {
      speak(answerText);
    }
  };

  return (
    <div className="p-5 bg-white border border-slate-200 rounded-xl shadow-sm transition-shadow hover:shadow-md">
      {/* Question with audio playback */}
      <div className="flex justify-between items-start gap-4 mb-3">
        <div className="flex-1">
          <h4 className="font-bold text-lg text-slate-800">{qaPair.question}</h4>
        </div>
        <button
          onClick={handleTogglePlayQuestion}
          className="flex-shrink-0 p-2 bg-indigo-100 text-indigo-600 rounded-full hover:bg-indigo-200 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          aria-label={isQuestionSpeaking ? "Stop question playback" : "Play question"}
        >
          {isQuestionSpeaking ? (
            <StopIcon className="w-4 h-4" />
          ) : (
            <PlayIcon className="w-4 h-4" />
          )}
        </button>
      </div>
      
      {/* Answer with audio playback */}
      <div className="flex justify-between items-start gap-4">
        <div className="flex-1">
          <p className="text-slate-600 leading-relaxed">
            {answerText}
          </p>
        </div>
        <button
          onClick={handleTogglePlayAnswer}
          className="flex-shrink-0 mt-1 p-2.5 bg-indigo-100 text-indigo-600 rounded-full hover:bg-indigo-200 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          aria-label={isAnswerSpeaking ? "Stop answer playback" : "Play answer"}
        >
          {isAnswerSpeaking ? (
            <StopIcon className="w-5 h-5" />
          ) : (
            <PlayIcon className="w-5 h-5" />
          )}
        </button>
      </div>

      {qaPair.claims && qaPair.claims.length > 0 && (
        <div className="mt-4">
          <FactCheckDisplay claims={qaPair.claims} />
        </div>
      )}
    </div>
  );
};

export default QACard;
