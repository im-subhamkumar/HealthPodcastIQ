import React, { useState, useCallback, useEffect } from 'react';
import { AppState, SummaryResult, EpisodeSequenceResult, HistoryItem } from './types';
import { processPodcast, createEpisodeSequence, checkBackendHealth, getHistory } from './services/summarizerService';
import InputForm from './components/InputForm';
import ProcessingView from './components/ProcessingView';
import ResultsDisplay from './components/ResultsDisplay';
import EpisodeSequenceDisplay from './components/EpisodeSequenceDisplay';
import { SettingsProvider } from './contexts/SettingsContext';

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [summaryResult, setSummaryResult] = useState<SummaryResult | null>(null);
  const [sequenceResult, setSequenceResult] = useState<EpisodeSequenceResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'offline'>('checking');
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const fetchHistory = useCallback(async () => {
    const data = await getHistory();
    setHistory(data);
  }, []);

  useEffect(() => {
    checkBackendHealth().then(isOnline => {
      setBackendStatus(isOnline ? 'connected' : 'offline');
      if (isOnline) fetchHistory();
    });
  }, [fetchHistory]);

  const handleProcess = useCallback(async (source: string | File | string[]) => {
    setAppState(AppState.PROCESSING);
    setSummaryResult(null);
    setSequenceResult(null);
    setErrorMessage('');

    try {
      if (Array.isArray(source)) {
        const result = await createEpisodeSequence(source);
        setSequenceResult(result);
      } else {
        const result = await processPodcast(source);
        setSummaryResult(result);
      }
      setAppState(AppState.SUCCESS);
    } catch (error) {
      let message = 'An unknown error occurred.';

      if (error instanceof Error) {
        message = error.message;

        // Provide user-friendly error messages
        if (message.includes('model not found') || message.includes('Model')) {
          message = `Model Error: ${message}\n\nPlease ensure the AI models are properly installed.`;
        } else if (message.includes('transcription') || message.includes('Whisper')) {
          message = `Transcription Error: ${message}\n\nPlease check your audio file format.`;
        } else if (message.includes('API') || message.includes('GEMINI')) {
          message = `API Error: ${message}\n\nPlease check your API key configuration.`;
        } else if (message.includes('Backend') || message.includes('connection')) {
          message = `Connection Error: ${message}\n\nPlease ensure the backend server is running.`;
        }
      } else if (typeof error === 'string') {
        message = error;
      }

      console.error('Processing error:', error);
      setErrorMessage(message);
      setAppState(AppState.ERROR);
    }
  }, []);

  const handleReset = useCallback(() => {
    setAppState(AppState.IDLE);
    setSummaryResult(null);
    setSequenceResult(null);
    setErrorMessage('');
  }, []);

  return (
    <SettingsProvider>
      <div className="min-h-screen bg-slate-50 font-sans flex items-center justify-center p-4">
        <div className="w-full max-w-4xl mx-auto">
          <header className="text-center mb-10">
            <h1 className="text-5xl font-bold text-slate-800">
              HealthPodcast<span className="text-indigo-600">IQ</span>
            </h1>
            <p className="text-slate-500 mt-2 text-sm">
              AI-Driven Podcast Intelliigence System
            </p>
          </header>

          <main className="bg-white border border-slate-200 rounded-2xl shadow-lg p-6 sm:p-8 transition-all duration-500 min-h-[300px]">
            {appState === AppState.IDLE && (
              <div className="space-y-8">
                <InputForm onProcess={handleProcess} />

                {history.length > 0 && (
                  <div className="pt-6 border-t border-slate-100">
                    <h3 className="text-lg font-bold text-slate-700 mb-4 flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                      </svg>
                      Recent History
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {history.slice(0, 4).map((item) => (
                        <div
                          key={item.uploadid}
                          onClick={() => handleProcess(item.source)}
                          className="flex items-center gap-3 p-3 rounded-xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/30 transition-all cursor-pointer group"
                        >
                          {item.thumbnail_url ? (
                            <img src={item.thumbnail_url} alt={item.title} className="w-16 h-12 object-cover rounded-lg shadow-sm" />
                          ) : (
                            <div className="w-16 h-12 bg-slate-100 rounded-lg flex items-center justify-center">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            </div>
                          )}
                          <div className="min-w-0 flex-grow">
                            <h4 className="text-sm font-semibold text-slate-800 truncate group-hover:text-indigo-600 transition-colors">{item.title}</h4>
                            <p className="text-xs text-slate-500">{new Date(item.created_at).toLocaleDateString()}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            {appState === AppState.PROCESSING && <ProcessingView />}
            {appState === AppState.SUCCESS && summaryResult && (
              <ResultsDisplay result={summaryResult} onReset={handleReset} />
            )}
            {appState === AppState.SUCCESS && sequenceResult && (
              <EpisodeSequenceDisplay result={sequenceResult} onReset={handleReset} />
            )}
            {appState === AppState.ERROR && (
              <div className="text-center">
                <h3 className="text-xl font-semibold text-red-500 mb-2">Processing Failed</h3>
                <p className="mt-2 text-slate-600 whitespace-pre-line">{errorMessage}</p>
                <button
                  onClick={handleReset}
                  className="mt-6 bg-purple-500 text-white font-semibold py-2 px-6 rounded-lg hover:bg-purple-600 transition-colors shadow-md"
                >
                  Try Again
                </button>
              </div>
            )}
          </main>

          <footer className="text-center mt-8 text-sm text-slate-500">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-200/50 border border-slate-200">
              <div className={`w-2 h-2 rounded-full ${backendStatus === 'connected' ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>
              <span className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
                {backendStatus === 'checking' ? 'System Check...' :
                  backendStatus === 'connected' ? 'Backend Connected' : 'Backend Offline'}
              </span>
            </div>
          </footer>
        </div>
      </div>
    </SettingsProvider>
  );
};

export default App;