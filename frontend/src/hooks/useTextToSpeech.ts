import { useState, useEffect, useCallback, useContext } from 'react';
import { SettingsContext } from '../contexts/SettingsContext';
import { VoiceOption } from '../types';

export const useTextToSpeech = () => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentlySpeakingText, setCurrentlySpeakingText] = useState<string | null>(null);
  const { voice: selectedVoice } = useContext(SettingsContext);

  // We don't strictly need to keep availableVoices in state for the logic to work,
  // but we do need to ensure voices are loaded by the browser.
  useEffect(() => {
    const loadVoices = () => {
      window.speechSynthesis.getVoices();
    };
    loadVoices();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);

  const speak = useCallback((text: string) => {
    if (!text || !window.speechSynthesis) return;

    // Always cancel current speech before starting new
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    
    let targetVoice: SpeechSynthesisVoice | undefined;

    // Helper for fuzzy matching voice names/langs
    const matches = (v: SpeechSynthesisVoice, terms: string[]) => 
        terms.every(term => v.name.toLowerCase().includes(term.toLowerCase()) || v.lang.toLowerCase().includes(term.toLowerCase()));

    if (selectedVoice === VoiceOption.FEMALE) {
        // Priority 1: Indian Female (Specific high-quality names on Mac/Chrome)
        targetVoice = voices.find(v => matches(v, ['en-IN', 'female'])) || 
                      voices.find(v => matches(v, ['Veena'])) ||     // Mac Indian Female
                      voices.find(v => matches(v, ['Sangeeta'])) ||  // Mac Indian Female
                      voices.find(v => matches(v, ['Google', 'en-IN'])); // Google India is usually female
        
        // Priority 2: International Female High Quality
        if (!targetVoice) {
            targetVoice = voices.find(v => matches(v, ['Google', 'US', 'English'])) || // Google US is female
                          voices.find(v => matches(v, ['Samantha'])) || // Mac US Female
                          voices.find(v => matches(v, ['female']));
        }
    } else {
        // Priority 1: Indian Male (Specific high-quality names on Mac/Chrome)
        targetVoice = voices.find(v => matches(v, ['en-IN', 'male'])) || 
                      voices.find(v => matches(v, ['Rishi'])) ||      // Mac Indian Male
                      voices.find(v => matches(v, ['Prabhat']));     // Another common Indian Male voice
        
        // Priority 2: International Male High Quality
        if (!targetVoice) {
            targetVoice = voices.find(v => matches(v, ['Google', 'UK', 'Male'])) ||
                          voices.find(v => matches(v, ['Daniel'])) || // Mac UK Male
                          voices.find(v => matches(v, ['Alex'])) ||   // Mac US Male
                          voices.find(v => matches(v, ['male']));
        }
    }

    // Default Fallback: Try to at least match the locale, otherwise first available
    if (!targetVoice) {
        targetVoice = voices.find(v => v.lang.includes('en-IN')) || voices[0];
    }

    if (targetVoice) {
        utterance.voice = targetVoice;
    }

    // Naturalness Tuning
    utterance.rate = 1.0;
    
    // User requested "Indian female low pitch". 
    // Standard pitch is 1.0. We lower it to 0.85 for female to sound more grounded/natural.
    // We also lower male pitch slightly to 0.9 for authority.
    utterance.pitch = selectedVoice === VoiceOption.MALE ? 0.9 : 0.85; 

    utterance.onstart = () => {
      setIsSpeaking(true);
      setCurrentlySpeakingText(text);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      setCurrentlySpeakingText(null);
    };
    
    utterance.onerror = (e) => {
        console.error("Speech Error:", e);
        setIsSpeaking(false);
        setCurrentlySpeakingText(null);
    };

    window.speechSynthesis.speak(utterance);
  }, [selectedVoice]);

  const stop = useCallback(() => {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      setCurrentlySpeakingText(null);
    }
  }, []);

  return { speak, stop, isSpeaking, currentlySpeakingText };
};