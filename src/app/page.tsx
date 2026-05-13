'use client';

import { useState, useEffect, useRef } from 'react';
import { Header } from '@/components/header';
import { Footer } from '@/components/footer';
import { StatementInput, ExtractionInput } from '@/components/statement-input';
import { StatementList } from '@/components/statement-list';
import { StatementEditor } from '@/components/statement-editor';
import { RelationshipGraph } from '@/components/relationship-graph';
import { RateLimitBanner } from '@/components/rate-limit-banner';
import { QuickStart } from '@/components/documentation';
import { LLMPrompts } from '@/components/llm-prompts';
import { Statement, UrlExtractionResult } from '@/lib/types';
import { getUserUuid } from '@/lib/user-uuid';
import { toast } from 'sonner';
import { ExportFormats } from '@/components/export-formats';
import { Network, FileText, BookOpen, Bot, Edit3, Eye, Code } from 'lucide-react';
import { HowItWorks, AboutCorpORate } from '@/components/about-sections';
import { CanonicalPredicates } from '@/components/canonical-predicates';
import { StatementTaxonomy } from '@/components/statement-taxonomy';
import { WarmUpDialog } from '@/components/warm-up-dialog';

// Show warm-up dialog after this many seconds of in-flight extraction
const WARMUP_DIALOG_THRESHOLD = 30;
// Poll cadence for async extraction runs. 2s is gentle on Vercel function
// invocations (~150/run cap for a 5-minute job) and tight enough that the
// UI updates promptly when the run finishes.
const POLL_INTERVAL_MS = 2_000;
// Hard ceiling on how long the browser will keep polling. Matches the
// Cerebrium replica's `response_grace_period` of 1h so a worst-case
// cold-start that does land successfully on the backend still reaches
// the user. The user can also hit Cancel to bail out before this fires.
const POLL_TIMEOUT_MS = 60 * 60 * 1000;

interface ExtractionPoll {
  run_id?: string;
  status?: 'pending' | 'running' | 'succeeded' | 'failed' | 'unknown';
  statements?: Statement[];
  cached?: boolean;
  message?: string;
  error?: string;
  inputText?: string;
  metadata?: UrlExtractionResult['metadata'];
  summary?: string;
}

async function pollExtractionRun(
  runId: string,
  signal: AbortSignal,
): Promise<ExtractionPoll> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new Error('aborted');
    const resp = await fetch(`/api/extract/status/${runId}`, { signal });
    if (!resp.ok && resp.status !== 200) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Status ${resp.status}`);
    }
    const data = (await resp.json()) as ExtractionPoll;
    if (data.status === 'succeeded') return data;
    if (data.status === 'failed') {
      throw new Error(data.error || 'Extraction failed on the backend');
    }
    // pending | running | unknown — wait and try again.
    await new Promise((resolve, reject) => {
      const t = setTimeout(resolve, POLL_INTERVAL_MS);
      signal.addEventListener('abort', () => {
        clearTimeout(t);
        reject(new Error('aborted'));
      }, { once: true });
    });
  }
  throw new Error('Extraction polling timed out');
}

export default function Home() {
  const [statements, setStatements] = useState<Statement[]>([]);
  const [editedStatements, setEditedStatements] = useState<Statement[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [rateLimitMessage, setRateLimitMessage] = useState<string | undefined>();
  const [userUuid, setUserUuid] = useState('');
  const [isLiking, setIsLiking] = useState(false);
  const [hasLiked, setHasLiked] = useState(false);
  const [showWarmUpDialog, setShowWarmUpDialog] = useState(false);
  const [showTimeoutDialog, setShowTimeoutDialog] = useState(false);
  const [warmUpDialogDismissed, setWarmUpDialogDismissed] = useState(false);
  // Held outside React state so the Cancel button can abort the in-flight
  // poll loop without re-rendering. Cleared once the run finishes.
  const pollAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setUserUuid(getUserUuid());
  }, []);

  // Fire a warmup ping on every page load. Cerebrium's cooldown is short
  // (60s) so the replica is almost always cold when a user lands; kicking
  // it via /api/warmup means the heavy model + index loads happen before
  // the user submits, not during. The route is fire-and-forget — it
  // doesn't write to Supabase or wait for a webhook, just nudges the
  // replica to spin up.
  useEffect(() => {
    fetch('/api/warmup', { method: 'POST', keepalive: true }).catch(() => {});
  }, []);

  // Show warm-up dialog after threshold seconds of loading
  useEffect(() => {
    if (isLoading && elapsedSeconds >= WARMUP_DIALOG_THRESHOLD && !warmUpDialogDismissed) {
      setShowWarmUpDialog(true);
    }
  }, [isLoading, elapsedSeconds, warmUpDialogDismissed]);

  const handleExtract = async (input: ExtractionInput) => {
    setIsLoading(true);
    setElapsedSeconds(0);
    setRateLimitMessage(undefined);
    setInputText(input.mode === 'text' ? input.text || '' : input.url || '');
    setHasLiked(false); // Reset like status for new extraction
    setShowWarmUpDialog(false);
    setShowTimeoutDialog(false);
    setWarmUpDialogDismissed(false);

    // Start elapsed time counter
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    try {
      // Build request body based on input mode
      const requestBody = input.mode === 'text'
        ? { text: input.text, useCanonicalPredicates: input.useCanonicalPredicates }
        : { url: input.url };

      const response = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok && response.status !== 202) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to extract statements');
      }

      // Submission response is either:
      //   - 202 with `{run_id, status:'pending'}` — async path; poll for result
      //   - 200 with the full result (cached, local-server, or cached-example
      //     fallback) — render directly without polling
      let payload = (await response.json()) as ExtractionPoll;
      if (payload.run_id && payload.status !== 'succeeded') {
        const ac = new AbortController();
        pollAbortRef.current = ac;
        try {
          payload = await pollExtractionRun(payload.run_id, ac.signal);
        } finally {
          pollAbortRef.current = null;
        }
      }

      const statements = payload.statements || [];
      setStatements(statements);
      setEditedStatements(JSON.parse(JSON.stringify(statements)));
      setHasChanges(false);
      setIsEditMode(false);

      if (payload.cached && payload.message) {
        setRateLimitMessage(payload.message);
        toast.warning(payload.message);
      } else if (statements.length === 0) {
        toast.info(input.mode === 'url' ? 'No statements found in the document' : 'No statements found in the text');
      } else {
        const suffix = input.mode === 'url' ? ' from document' : '';
        toast.success(`Extracted ${statements.length} statement${statements.length !== 1 ? 's' : ''}${suffix}`);
      }
    } catch (error) {
      console.error('Extraction error:', error);
      const message = error instanceof Error ? error.message : 'Failed to extract statements';
      // User-initiated cancel: swallow silently. The poll loop throws an
      // Error('aborted') when the AbortController fires; we don't want to
      // show a toast or the timeout dialog for that.
      if (/^aborted$/i.test(message)) {
        toast.info('Cancelled');
        return;
      }
      // Network-level timeout (poll exceeded its deadline) → show the timeout
      // dialog rather than a bare toast.
      const looksLikeTimeout = /timeout|504|gateway/i.test(message);
      if (looksLikeTimeout) {
        setShowTimeoutDialog(true);
        return;
      }
      toast.error(message);
    } finally {
      clearInterval(timerInterval);
      setIsLoading(false);
      setElapsedSeconds(0);
      setShowWarmUpDialog(false);
      pollAbortRef.current = null;
    }
  };

  const handleCancelExtraction = () => {
    pollAbortRef.current?.abort();
  };

  const handleStatementsChange = (newStatements: Statement[]) => {
    setEditedStatements(newStatements);
    setHasChanges(JSON.stringify(newStatements) !== JSON.stringify(statements));
  };

  const handleSubmitCorrection = async () => {
    if (!inputText) {
      toast.error('No input text to submit');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch('/api/corrections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputText,
          statements: editedStatements,
          userUuid,
          source: 'correction',
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to submit correction');
      }

      toast.success('Correction submitted! Thank you for contributing.');
      setStatements(editedStatements);
      setHasChanges(false);
    } catch (error) {
      console.error('Submit error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to submit correction');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLike = async () => {
    if (!inputText || statements.length === 0) {
      return;
    }

    setIsLiking(true);

    try {
      const response = await fetch('/api/corrections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputText,
          statements,
          userUuid,
          source: 'liked',
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to save');
      }

      setHasLiked(true);
      toast.success('Thanks for the feedback!');
    } catch (error) {
      console.error('Like error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to save');
    } finally {
      setIsLiking(false);
    }
  };

  const toggleEditMode = () => {
    if (isEditMode) {
      // Switching to view mode - reset changes if any
      setEditedStatements(JSON.parse(JSON.stringify(statements)));
      setHasChanges(false);
    }
    setIsEditMode(!isEditMode);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 border-b">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <span className="section-label">corp-extractor demo</span>
              <h1 className="text-4xl md:text-5xl font-black mt-4 tracking-tight">
                EXTRACT STATEMENTS.
                <br />
                <span className="text-gray-400">MAP RELATIONSHIPS.</span>
              </h1>
              <p className="mt-4 text-gray-600 max-w-2xl mx-auto">
                A Python library designed to analyze complex text and extract relationship information
                about people and organizations. Runs entirely on your hardware (RTX 4090+, Apple M1 16GB+)
                with no external API dependencies. Uses fine-tuned{' '}
                <a
                  href="https://blog.google/technology/developers/t5gemma-2/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-red-600 hover:underline font-medium"
                >
                  T5-Gemma 2
                </a>{' '}
                for statement splitting and coreference resolution, plus{' '}
                <a
                  href="https://github.com/urchade/GLiNER"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-red-600 hover:underline font-medium"
                >
                  GLiNER2
                </a>{' '}
                for entity extraction. Includes a database of 9.7M+ organizations and 63M+ people
                with USearch HNSW indexes for fast entity qualification (~100GB disk for all models and data).
              </p>
            </div>

            {/* Input Section */}
            <div className="brutal-card p-6 md:p-8">
              <StatementInput onExtract={handleExtract} onCancel={handleCancelExtraction} isLoading={isLoading} elapsedSeconds={elapsedSeconds} />
            </div>
          </div>
        </section>

        {/* Results Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            <RateLimitBanner message={rateLimitMessage} />

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Statements List / Editor */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-red-600" />
                    <h2 className="font-bold text-xl">Statements</h2>
                  </div>
                  {statements.length > 0 && (
                    <button
                      onClick={toggleEditMode}
                      className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-semibold text-gray-600 hover:text-black border rounded transition-colors"
                    >
                      {isEditMode ? (
                        <>
                          <Eye className="w-4 h-4" />
                          View
                        </>
                      ) : (
                        <>
                          <Edit3 className="w-4 h-4" />
                          Correct
                        </>
                      )}
                    </button>
                  )}
                </div>
                <div className="editorial-card p-4 md:p-6 min-h-[400px]">
                  {isEditMode ? (
                    <StatementEditor
                      statements={editedStatements}
                      onChange={handleStatementsChange}
                      onSubmit={handleSubmitCorrection}
                      isSubmitting={isSubmitting}
                      hasChanges={hasChanges}
                    />
                  ) : (
                    <StatementList
                      statements={statements}
                      onLike={handleLike}
                      isLiking={isLiking}
                      hasLiked={hasLiked}
                    />
                  )}
                </div>
              </div>

              {/* Relationship Graph */}
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <Network className="w-5 h-5 text-red-600" />
                  <h2 className="font-bold text-xl">Relationship Graph</h2>
                </div>
                <RelationshipGraph statements={isEditMode ? editedStatements : statements} />
              </div>
            </div>

            {/* Export Formats */}
            <div className="mt-8">
              <div className="flex items-center gap-2 mb-4">
                <Code className="w-5 h-5 text-red-600" />
                <h2 className="font-bold text-xl">Export</h2>
              </div>
              <ExportFormats statements={isEditMode ? editedStatements : statements} />
            </div>

            {/* Canonical Predicates Reference */}
            <div className="mt-8">
              <CanonicalPredicates />
            </div>

            {/* Statement Taxonomy Reference */}
            <div className="mt-8">
              <StatementTaxonomy />
            </div>
          </div>
        </section>

        {/* Quick Start Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8 bg-gray-50/50">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <BookOpen className="w-5 h-5 text-red-600" />
              <h2 className="font-bold text-xl">Quick Start</h2>
            </div>
            <QuickStart />
          </div>
        </section>

        {/* LLM Prompts Section */}
        <section className="py-12 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center gap-2 mb-6">
              <Bot className="w-5 h-5 text-red-600" />
              <h2 className="font-bold text-xl">For AI Assistants</h2>
            </div>
            <LLMPrompts />
          </div>
        </section>

        {/* About Section */}
        <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50 border-t">
          <div className="max-w-4xl mx-auto text-center">
            <span className="section-label">THE PIPELINE</span>
            <h2 className="text-2xl md:text-3xl font-black mt-4">
              Multiple Models, One Pipeline
            </h2>
            <div className="mt-6 text-gray-600 space-y-4 text-left max-w-2xl mx-auto">
              <p>
                Corp-extractor uses multiple fine-tuned small models to transform unstructured text into
                structured relationship data—all running locally on your hardware with no external services.
              </p>
              <p>
                <strong>Pipeline stages:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li><strong>T5-Gemma 2</strong> (540M params) — Splits text into atomic statements and resolves coreferences. Trained on 70,000+ pages of corporate and news documents.</li>
                <li><strong>GLiNER2</strong> (205M params) — Extracts subject/predicate/object with entity types (ORG, PERSON, GPE, etc.) and 324 predefined predicates.</li>
                <li><strong>Entity Database</strong> — Qualifies entities against 9.7M+ organizations and 63M+ people with USearch HNSW indexes for sub-millisecond lookups.</li>
                <li><strong>BERT classifiers</strong> — Small models for sentiment labeling and embedding similarity for taxonomy classification.</li>
              </ul>
              <p>
                <strong>Hardware:</strong> Requires ~100GB disk for all models and database. Runs on RTX 4090+ or Apple M1/M2/M3 with 16GB+ RAM.
              </p>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <HowItWorks />

        {/* About Corp-o-Rate Section */}
        <AboutCorpORate />
      </main>

      <Footer />

      {/* Warm-up Dialog - shown after 30 seconds of loading */}
      <WarmUpDialog
        isOpen={showWarmUpDialog}
        elapsedSeconds={elapsedSeconds}
        onClose={() => {
          setShowWarmUpDialog(false);
          setWarmUpDialogDismissed(true);
        }}
      />

      {/* Timeout Dialog - shown when request times out */}
      <WarmUpDialog
        isOpen={showTimeoutDialog}
        elapsedSeconds={elapsedSeconds}
        onClose={() => setShowTimeoutDialog(false)}
        isTimeout
      />
    </div>
  );
}
