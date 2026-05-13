'use client';

import { useState, useRef, useCallback } from 'react';
import { Loader2, Sparkles, Trash2, Shuffle, Link, FileText, Upload, X, File, XCircle } from 'lucide-react';
import { CACHED_INPUT } from '@/lib/cached-example';
import { toast } from 'sonner';

type InputMode = 'text' | 'url';

interface ExtractionInput {
  mode: InputMode;
  text?: string;
  url?: string;
  fileName?: string;
  useCanonicalPredicates?: boolean;
}

// Supported file types for text extraction
const TEXT_FILE_TYPES = ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm'];
const MAX_FILE_SIZE = 1024 * 1024; // 1MB

interface StatementInputProps {
  onExtract: (input: ExtractionInput) => void;
  /** Called when the user clicks the Cancel button while a run is in-flight.
   *  Should abort the polling loop on the page-level controller. */
  onCancel?: () => void;
  isLoading: boolean;
  elapsedSeconds?: number;
}

const EXAMPLE_TEXTS = [
  {
    label: 'Corporate ESG',
    text: CACHED_INPUT,
  },
  {
    label: 'News Article',
    text: `Amazon announced plans to hire 150,000 workers for the holiday season. The company's CEO Andy Jassy said the positions would include warehouse and delivery roles across the United States. Labor unions criticized the move, citing concerns about working conditions. The announcement comes as retail giants prepare for increased consumer spending during Black Friday and Cyber Monday.`,
  },
  {
    label: 'Scientific',
    text: `Researchers at MIT have developed a new battery technology that could triple the range of electric vehicles. The team, led by Professor Jennifer Chen, published their findings in Nature Energy. Tesla has expressed interest in licensing the technology. The breakthrough uses solid-state lithium cells instead of traditional liquid electrolytes, significantly improving energy density.`,
  },
];

export function StatementInput({ onExtract, onCancel, isLoading, elapsedSeconds = 0 }: StatementInputProps) {
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState<{ name: string; content: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [useCanonicalPredicates, setUseCanonicalPredicates] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileRead = useCallback(async (selectedFile: File) => {
    // Check file size
    if (selectedFile.size > MAX_FILE_SIZE) {
      toast.error('File too large. Maximum size is 1MB.');
      return;
    }

    // Check file type
    const fileName = selectedFile.name.toLowerCase();
    const isTextFile = TEXT_FILE_TYPES.some(ext => fileName.endsWith(ext));

    if (!isTextFile) {
      toast.error('Unsupported file type. Please use a text file (.txt, .md, .csv, .json, .xml, .html) or enter a URL for PDFs.');
      return;
    }

    try {
      const content = await selectedFile.text();
      if (!content.trim()) {
        toast.error('File is empty');
        return;
      }
      setFile({ name: selectedFile.name, content });
      setUrl(''); // Clear URL when file is selected
      toast.success(`File loaded: ${selectedFile.name}`);
    } catch (error) {
      console.error('File read error:', error);
      toast.error('Failed to read file');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileRead(droppedFile);
    }
  }, [handleFileRead]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileRead(selectedFile);
    }
  }, [handleFileRead]);

  const clearFile = useCallback(() => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMode === 'text' && text.trim() && !isLoading) {
      onExtract({ mode: 'text', text: text.trim(), useCanonicalPredicates });
    } else if (inputMode === 'url' && !isLoading) {
      // Check if we have a file or URL
      if (file) {
        // Send file content as text
        onExtract({ mode: 'text', text: file.content, fileName: file.name, useCanonicalPredicates });
      } else if (url.trim()) {
        // Validate URL
        try {
          new URL(url.trim());
          onExtract({ mode: 'url', url: url.trim(), useCanonicalPredicates });
        } catch {
          toast.error('Please enter a valid URL');
        }
      }
    }
  };

  const loadExample = (exampleText: string) => {
    setInputMode('text');
    setText(exampleText);
  };

  const generateRandomText = async () => {
    setIsGenerating(true);
    setText(''); // Clear existing text before streaming
    try {
      const response = await fetch('/api/generate', { method: 'POST' });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to generate text');
      }

      // Read the stream
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let streamedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        streamedText += chunk;
        setText(streamedText);
      }

      toast.success('Random text generated!');
    } catch (error) {
      console.error('Generation error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate text');
    } finally {
      setIsGenerating(false);
    }
  };

  const clearInput = () => {
    if (inputMode === 'text') {
      setText('');
    } else {
      setUrl('');
      clearFile();
    }
  };

  const charCount = text.length;
  const maxChars = 4000;
  const hasInput = inputMode === 'text' ? text.trim() : (url.trim() || file);

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit}>
        {/* Input mode toggle */}
        <div className="mb-4 flex items-center gap-1 p-1 bg-gray-100 rounded-lg w-fit">
          <button
            type="button"
            onClick={() => setInputMode('text')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              inputMode === 'text'
                ? 'bg-white text-black shadow-sm'
                : 'text-gray-600 hover:text-black'
            }`}
            disabled={isLoading}
          >
            <FileText className="w-4 h-4" />
            Text
          </button>
          <button
            type="button"
            onClick={() => setInputMode('url')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              inputMode === 'url'
                ? 'bg-white text-black shadow-sm'
                : 'text-gray-600 hover:text-black'
            }`}
            disabled={isLoading}
          >
            <Link className="w-4 h-4" />
            URL / Document
          </button>
        </div>

        {inputMode === 'text' ? (
          <>
            {/* Example buttons */}
            <div className="mb-4 flex flex-wrap gap-2">
              <span className="text-sm text-gray-500 mr-2 self-center">Try an example:</span>
              {EXAMPLE_TEXTS.map((example) => (
                <button
                  key={example.label}
                  type="button"
                  onClick={() => loadExample(example.text)}
                  className="px-3 py-1.5 text-sm font-medium border border-gray-200 hover:border-black transition-colors"
                  disabled={isLoading || isGenerating}
                >
                  {example.label}
                </button>
              ))}
              <button
                type="button"
                onClick={generateRandomText}
                className="px-3 py-1.5 text-sm font-medium border border-gray-300 bg-gray-800 text-white hover:bg-gray-700 transition-colors inline-flex items-center gap-1.5"
                disabled={isLoading || isGenerating}
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 spinner" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Shuffle className="w-3.5 h-3.5" />
                    Randomly Generated
                  </>
                )}
              </button>
            </div>

            {/* Textarea */}
            <div className="relative">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your text here to extract statements..."
                className="input-textarea min-h-[200px]"
                maxLength={maxChars}
                disabled={isLoading}
              />

              {/* Character count */}
              <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                {charCount.toLocaleString()} / {maxChars.toLocaleString()}
              </div>
            </div>
          </>
        ) : (
          <>
            {/* File upload area */}
            <div className="mb-4">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.md,.csv,.json,.xml,.html,.htm"
                onChange={handleFileSelect}
                className="hidden"
                disabled={isLoading}
              />

              {file ? (
                // File selected - show file info
                <div className="flex items-center justify-between p-4 border-2 border-green-500 bg-green-50">
                  <div className="flex items-center gap-3">
                    <File className="w-5 h-5 text-green-600" />
                    <div>
                      <div className="font-medium text-green-800">{file.name}</div>
                      <div className="text-sm text-green-600">
                        {file.content.length.toLocaleString()} characters
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={clearFile}
                    className="p-2 text-green-600 hover:text-green-800 hover:bg-green-100 rounded transition-colors"
                    disabled={isLoading}
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              ) : (
                // Drop zone
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex flex-col items-center justify-center p-8 border-2 border-dashed cursor-pointer transition-colors ${
                    isDragging
                      ? 'border-red-500 bg-red-50'
                      : 'border-gray-300 hover:border-gray-400 bg-gray-50'
                  }`}
                >
                  <Upload className={`w-8 h-8 mb-2 ${isDragging ? 'text-red-500' : 'text-gray-400'}`} />
                  <p className="text-sm font-medium text-gray-700">
                    Drop a file here or click to upload
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supports .txt, .md, .csv, .json, .xml, .html (max 1MB)
                  </p>
                </div>
              )}
            </div>

            {/* Divider */}
            {!file && (
              <div className="flex items-center gap-4 mb-4">
                <div className="flex-1 h-px bg-gray-200" />
                <span className="text-sm text-gray-400">or enter a URL</span>
                <div className="flex-1 h-px bg-gray-200" />
              </div>
            )}

            {/* URL input */}
            {!file && (
              <div className="mb-2">
                <p className="text-sm text-gray-500 mb-3">
                  Enter a URL to extract statements from web articles or PDF documents.
                </p>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/article or https://example.com/document.pdf"
                  className="w-full px-4 py-3 border-2 border-gray-200 focus:border-black focus:outline-none transition-colors text-base"
                  disabled={isLoading}
                />
              </div>
            )}
          </>
        )}

        {/* Options */}
        <div className="mt-4 flex items-center gap-3">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useCanonicalPredicates}
              onChange={(e) => setUseCanonicalPredicates(e.target.checked)}
              disabled={isLoading}
              className="w-4 h-4 accent-red-600 cursor-pointer"
            />
            <span className="text-sm text-gray-700">
              Use default predicates
            </span>
            <a
              href="#canonical-predicates"
              className="text-xs text-gray-400 hover:text-red-600 transition-colors"
              title="View list of default predicates"
            >
              (view list)
            </a>
          </label>
        </div>

        {/* Actions */}
        <div className="mt-4 flex flex-wrap gap-3 items-center justify-between">
          <div className="flex gap-3">
            {isLoading ? (
              <button
                type="button"
                onClick={onCancel}
                disabled={!onCancel}
                className="btn-brutal"
                aria-label="Cancel extraction"
                title="Stop polling for results — Cerebrium keeps running but the UI lets go."
              >
                <Loader2 className="w-4 h-4 mr-2 spinner" />
                <span className="mr-2">
                  {elapsedSeconds > 0 ? `Processing... (${elapsedSeconds}s)` : 'Submitting...'}
                </span>
                <XCircle className="w-4 h-4" />
                <span className="ml-1">Cancel</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!hasInput}
                className="btn-brutal"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                {inputMode === 'url'
                  ? (file ? 'Extract from File' : 'Extract from URL')
                  : 'Extract Statements'}
              </button>
            )}

            {hasInput && (
              <button
                type="button"
                onClick={clearInput}
                className="btn-secondary"
                disabled={isLoading}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear
              </button>
            )}
          </div>

          <p className="text-sm text-gray-500">
            Powered by{' '}
            <a
              href="https://corp-o-rate.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-red-600 hover:underline"
            >
              corp-o-rate
            </a>
          </p>
        </div>
      </form>
    </div>
  );
}

// Export types for use in other components
export type { ExtractionInput, InputMode };
