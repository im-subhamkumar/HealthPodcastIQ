import React, { createContext, useState, ReactNode } from 'react';
import { SummaryLength, VoiceOption } from '../types';

interface SettingsContextType {
  length: SummaryLength;
  setLength: (length: SummaryLength) => void;
  voice: VoiceOption;
  setVoice: (voice: VoiceOption) => void;
}

export const SettingsContext = createContext<Partial<SettingsContextType>>({});

interface SettingsProviderProps {
  children: ReactNode;
}

export const SettingsProvider: React.FC<SettingsProviderProps> = ({ children }) => {
  const [length, setLength] = useState<SummaryLength>(SummaryLength.MEDIUM);
  const [voice, setVoice] = useState<VoiceOption>(VoiceOption.FEMALE);

  return (
    <SettingsContext.Provider value={{ length, setLength, voice, setVoice }}>
      {children}
    </SettingsContext.Provider>
  );
};
