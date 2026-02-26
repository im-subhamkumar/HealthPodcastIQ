import React, { useState, useCallback, useRef } from 'react';
import { LinkIcon, UploadIcon } from './icons';

type InputType = 'url' | 'file';
type Mode = 'summary' | 'sequence';

interface InputFormProps {
  onProcess: (source: string | File | string[] | File[]) => void;
}

const InputForm: React.FC<InputFormProps> = ({ onProcess }) => {
  const [mode, setMode] = useState<Mode>('summary');
  const [inputType, setInputType] = useState<InputType>('url');
  const [sequenceInputType, setSequenceInputType] = useState<InputType>('url');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [sequenceFiles, setSequenceFiles] = useState<File[]>([]);
  const [sequenceUrls, setSequenceUrls] = useState<string[]>(['', '']);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sequenceFilesInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSequenceFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSequenceFiles(Array.from(e.target.files));
    }
  };

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUrl(e.target.value);
  };

  const handleSequenceUrlChange = (index: number, value: string) => {
    const newUrls = [...sequenceUrls];
    newUrls[index] = value;
    setSequenceUrls(newUrls);
  };

  const addSequenceUrlInput = () => setSequenceUrls([...sequenceUrls, '']);
  const removeSequenceUrlInput = (index: number) => setSequenceUrls(sequenceUrls.filter((_, i) => i !== index));
  const removeSequenceFile = (index: number) => setSequenceFiles(sequenceFiles.filter((_, i) => i !== index));

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'summary') {
      if (inputType === 'url' && url) onProcess(url);
      else if (inputType === 'file' && file) onProcess(file);
    } else {
      if (sequenceInputType === 'url') {
        const validUrls = sequenceUrls.filter(u => u.trim() !== '');
        if (validUrls.length > 1) onProcess(validUrls);
      } else {
        if (sequenceFiles.length > 1) onProcess(sequenceFiles);
      }
    }
  }, [mode, inputType, sequenceInputType, url, file, sequenceFiles, sequenceUrls, onProcess]);

  const triggerFileSelect = () => fileInputRef.current?.click();
  const triggerSequenceFilesSelect = () => sequenceFilesInputRef.current?.click();

  const isSummarySubmitDisabled = (inputType === 'url' && !url) || (inputType === 'file' && !file);
  const isSequenceSubmitDisabled = sequenceInputType === 'url'
    ? sequenceUrls.filter(u => u.trim() !== '').length < 2
    : sequenceFiles.length < 2;
  const isSubmitDisabled = mode === 'summary' ? isSummarySubmitDisabled : isSequenceSubmitDisabled;

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Mode Toggle - Updated with purple/indigo theme */}
      <div className="flex justify-center p-1 bg-slate-100 rounded-lg">
        <button type="button" onClick={() => setMode('summary')} className={`w-1/2 px-4 py-2 rounded-md text-sm font-semibold transition-colors ${mode === 'summary' ? 'bg-white shadow text-purple-600' : 'text-slate-500'}`}>
          Single Summary
        </button>
        <button type="button" onClick={() => setMode('sequence')} className={`w-1/2 px-4 py-2 rounded-md text-sm font-semibold transition-colors ${mode === 'sequence' ? 'bg-white shadow text-purple-600' : 'text-slate-500'}`}>
          Episode Sequencing
        </button>
      </div>

      {mode === 'summary' ? (
        <div className="animate-fade-in">
          <div className="flex bg-slate-100/80 rounded-t-lg p-1">
            <button type="button" onClick={() => setInputType('url')} className={`w-full py-2.5 rounded-md transition-colors text-sm font-semibold ${inputType === 'url' ? 'bg-white shadow text-purple-600' : 'text-slate-500 hover:bg-slate-200/50'}`}>
              <span className="flex items-center justify-center gap-2"><LinkIcon className="w-4 h-4" /> YouTube Link</span>
            </button>
            <button type="button" onClick={() => setInputType('file')} className={`w-full py-2.5 rounded-md transition-colors text-sm font-semibold ${inputType === 'file' ? 'bg-white shadow text-purple-600' : 'text-slate-500 hover:bg-slate-200/50'}`}>
              <span className="flex items-center justify-center gap-2"><UploadIcon className="w-4 h-4" /> Upload File</span>
            </button>
          </div>

          <div className="p-4 border border-t-0 border-slate-200 rounded-b-lg">
            {inputType === 'url' && (
              <div className="animate-fade-in">
                <label htmlFor="podcast-url" className="text-sm font-semibold text-slate-700 mb-2 block">Enter podcast YouTube link</label>
                <input id="podcast-url" type="text" value={url} onChange={handleUrlChange} placeholder="https://www.youtube.com/watch?v=..." className="w-full px-4 py-2.5 bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-400 transition-shadow outline-none text-slate-900" />
              </div>
            )}
            {inputType === 'file' && (
              <div className="animate-fade-in text-center">
                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="audio/*,video/*" />
                <p className="text-sm text-slate-600 mb-3">{file ? `Selected file:` : 'Select an audio or video file to upload.'}</p>
                <button type="button" onClick={triggerFileSelect} className="w-full max-w-xs mx-auto flex justify-center items-center gap-2 px-4 py-2.5 border border-slate-300 rounded-lg text-slate-700 font-semibold bg-white hover:bg-slate-50 transition">
                  {file ? file.name : 'Choose File'}
                </button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="animate-fade-in">
          <div className="flex bg-slate-100/80 rounded-t-lg p-1">
            <button type="button" onClick={() => setSequenceInputType('url')} className={`w-full py-2.5 rounded-md transition-colors text-sm font-semibold ${sequenceInputType === 'url' ? 'bg-white shadow text-purple-600' : 'text-slate-500 hover:bg-slate-200/50'}`}>
              <span className="flex items-center justify-center gap-2"><LinkIcon className="w-4 h-4" /> YouTube Links</span>
            </button>
            <button type="button" onClick={() => setSequenceInputType('file')} className={`w-full py-2.5 rounded-md transition-colors text-sm font-semibold ${sequenceInputType === 'file' ? 'bg-white shadow text-purple-600' : 'text-slate-500 hover:bg-slate-200/50'}`}>
              <span className="flex items-center justify-center gap-2"><UploadIcon className="w-4 h-4" /> Upload Files</span>
            </button>
          </div>

          <div className="p-4 border border-t-0 border-slate-200 rounded-b-lg space-y-4">
            {sequenceInputType === 'url' && (
              <div className="animate-fade-in space-y-4">
                <p className="text-center text-sm text-slate-600">Enter at least two YouTube links to sequence episodes.</p>
                {sequenceUrls.map((u, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className="relative flex-grow">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><LinkIcon className="h-5 w-5 text-slate-400" /></div>
                      <input type="text" value={u} onChange={(e) => handleSequenceUrlChange(index, e.target.value)} placeholder={`YouTube Podcast Link #${index + 1}`} className="w-full pl-10 pr-4 py-3 bg-white border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-400 transition-shadow outline-none text-slate-900" />
                    </div>
                    {sequenceUrls.length > 2 && (
                      <button type="button" onClick={() => removeSequenceUrlInput(index)} className="p-2 text-slate-400 hover:text-red-500 rounded-full transition" aria-label="Remove link">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                      </button>
                    )}
                  </div>
                ))}
                <button type="button" onClick={addSequenceUrlInput} className="w-full text-sm font-semibold text-purple-600 hover:text-purple-700 transition-colors">+ Add another link</button>
              </div>
            )}
            {sequenceInputType === 'file' && (
              <div className="animate-fade-in text-center space-y-4">
                <input type="file" ref={sequenceFilesInputRef} onChange={handleSequenceFilesChange} className="hidden" accept="audio/*,video/*" multiple />
                <p className="text-sm text-slate-600">Select two or more audio or video files to sequence.</p>

                <div className="space-y-2">
                  {sequenceFiles.map((f, index) => (
                    <div key={index} className="flex items-center justify-between px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg">
                      <span className="text-sm text-slate-700 truncate max-w-[200px]">{f.name}</span>
                      <button type="button" onClick={() => removeSequenceFile(index)} className="text-slate-400 hover:text-red-500 transition">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                      </button>
                    </div>
                  ))}
                </div>

                <button type="button" onClick={triggerSequenceFilesSelect} className="w-full flex justify-center items-center gap-2 px-4 py-2.5 border border-slate-300 rounded-lg text-slate-700 font-semibold bg-white hover:bg-slate-50 transition">
                  <UploadIcon className="w-4 h-4" /> {sequenceFiles.length > 0 ? 'Add More Files' : 'Choose Files'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      <div className="pt-2">
        <button
          type="submit"
          disabled={isSubmitDisabled}
          className="w-full bg-purple-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-purple-600 disabled:bg-slate-300 disabled:text-slate-500 transition-all transform hover:scale-[1.02] disabled:scale-100 shadow-md hover:shadow-lg"
        >
          {mode === 'summary' ? 'Generate Summary' : 'Sequence Episodes'}
        </button>
      </div>
    </form>
  );
};

export default InputForm;