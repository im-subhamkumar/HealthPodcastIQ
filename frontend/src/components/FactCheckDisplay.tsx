import React, { useState } from 'react';
import { Claim } from '../types';
import { CheckCircleIcon, XCircleIcon, InformationCircleIcon, ChevronDownIcon } from './icons';

interface FactCheckDisplayProps {
  claims: Claim[];
}

const FactCheckDisplay: React.FC<FactCheckDisplayProps> = ({ claims }) => {
  return (
    <div className="mt-4 pt-4 border-t border-slate-200">
      <h5 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
        FACT-CHECK
      </h5>
      <div className="space-y-3">
        {claims.map((claim, index) => (
          <ClaimItem key={index} claim={claim} />
        ))}
      </div>
    </div>
  );
};

const ClaimItem: React.FC<{ claim: Claim }> = ({ claim }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getVerdictStyles = () => {
    switch (claim.verdict) {
      case 'SUPPORTS':
        return {
          Icon: CheckCircleIcon,
          bgColor: 'bg-green-50/80',
          textColor: 'text-green-800',
          confidenceColor: 'text-green-600',
          progressBg: 'bg-green-200/50',
          progressColor: 'bg-green-500',
          evidenceBgColor: 'bg-green-100/70',
        };
      case 'REFUTES':
        return {
          Icon: XCircleIcon,
          bgColor: 'bg-red-50/80',
          textColor: 'text-red-800',
          confidenceColor: 'text-red-600',
          progressBg: 'bg-red-200/50',
          progressColor: 'bg-red-500',
          evidenceBgColor: 'bg-red-100/70',
        };
      default:
        return {
          Icon: InformationCircleIcon,
          bgColor: 'bg-slate-100/80',
          textColor: 'text-slate-800',
          confidenceColor: 'text-slate-600',
          progressBg: 'bg-slate-200/50',
          progressColor: 'bg-slate-500',
          evidenceBgColor: 'bg-slate-200/70',
        };
    }
  };

  const { Icon, bgColor, textColor, confidenceColor, progressBg, progressColor, evidenceBgColor } = getVerdictStyles();

  return (
    <div className={`p-4 rounded-lg ${bgColor} border border-opacity-20 ${claim.verdict === 'SUPPORTS' ? 'border-green-300' : claim.verdict === 'REFUTES' ? 'border-red-300' : 'border-slate-300'}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-shrink-0 pt-0.5">
           <Icon className={`w-6 h-6 ${confidenceColor}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-medium ${textColor} break-words`}>
             <span className="opacity-70 font-normal">Claim:</span> "{claim.claim}"
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm font-bold whitespace-nowrap flex-shrink-0">
          <span className={confidenceColor}>{claim.confidence}%</span>
          <div className={`w-16 h-2 ${progressBg} rounded-full overflow-hidden`}>
            <div 
              className={`h-full rounded-full transition-all duration-300 ${progressColor}`} 
              style={{ width: `${claim.confidence}%` }}
            ></div>
          </div>
        </div>
      </div>
       <div className="pl-10 mt-3">
         <button 
           onClick={() => setIsExpanded(!isExpanded)} 
           className="flex items-center text-xs font-semibold text-indigo-600 hover:text-indigo-700 transition-colors"
         >
            {isExpanded ? 'Hide Evidence' : 'Show Evidence'}
            <ChevronDownIcon className={`w-4 h-4 ml-1 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} />
         </button>
         {isExpanded && (
            <div className={`mt-2 p-3 rounded-md ${evidenceBgColor} border border-opacity-30 ${claim.verdict === 'SUPPORTS' ? 'border-green-200' : claim.verdict === 'REFUTES' ? 'border-red-200' : 'border-slate-200'}`}>
              <p className={`text-xs leading-relaxed ${textColor}`}>
                  {claim.explanation}
              </p>
            </div>
         )}
       </div>
    </div>
  );
};

export default FactCheckDisplay;
