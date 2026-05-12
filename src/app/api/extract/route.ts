import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { CACHED_EXAMPLE } from '@/lib/cached-example';
import { ExtractionResult } from '@/lib/types';
import { getCachedStatements, setCachedStatements } from '@/lib/cache';

// Cold-boot of a Cerebrium replica (T5-Gemma2 + GLiNER2 + GGUF qualifier
// loading from /persistent-storage) can take several minutes on the very
// first hit after scale-to-zero, and the full 5-stage pipeline (Stage 2
// GLiNER2 + Stage 3 qualification + Stage 5 taxonomy) is itself minute-scale
// on longer inputs. We use a wide window and a single retry path so the
// second attempt lands on a now-warm replica when the first cold attempt
// aborts. Requires the Vercel project plan to allow this maxDuration.
export const maxDuration = 800;

const CEREBRIUM_EXTRACT_URL = process.env.CEREBRIUM_EXTRACT_URL;
const CEREBRIUM_EXTRACT_URL_URL = process.env.CEREBRIUM_EXTRACT_URL_URL;
const CEREBRIUM_TOKEN = process.env.CEREBRIUM_TOKEN;
const LOCAL_MODEL_URL = process.env.LOCAL_MODEL_URL;

// Per-attempt client-side timeout. We give the first attempt the bulk of the
// budget and reserve enough for a single retry to land if the first aborts
// (typically a cold replica that's now warming). The two windows must fit
// inside `maxDuration` minus a few seconds of overhead.
const FIRST_ATTEMPT_TIMEOUT_MS = 720_000;
const RETRY_ATTEMPT_TIMEOUT_MS = 60_000;

interface CerebriumEnvelope {
  run_id?: string;
  result?: unknown;
  run_time_ms?: number;
}

/** Distinguish "the client gave up" from "the server returned an error". */
class CerebriumTimeoutError extends Error {
  constructor(public timeoutMs: number) {
    super(`Cerebrium request aborted after ${Math.round(timeoutMs / 1000)}s`);
    this.name = 'CerebriumTimeoutError';
  }
}
class CerebriumResponseError extends Error {
  constructor(public status: number, public body: string) {
    super(`Cerebrium returned status ${status}: ${body.slice(0, 300)}`);
    this.name = 'CerebriumResponseError';
  }
}

async function callCerebrium(endpointUrl: string, body: object, timeoutMs: number): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let response: Response;
    try {
      response = await fetch(endpointUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CEREBRIUM_TOKEN}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        throw new CerebriumTimeoutError(timeoutMs);
      }
      throw err;
    }
    if (!response.ok) {
      const errorText = await response.text().catch(() => '<no body>');
      throw new CerebriumResponseError(response.status, errorText);
    }
    const envelope = (await response.json()) as CerebriumEnvelope;
    return envelope.result ?? envelope;
  } finally {
    clearTimeout(timer);
  }
}

async function callCerebriumWithRetry(endpointUrl: string, body: object): Promise<unknown> {
  // First attempt — likely lands on a cold replica if no recent traffic.
  try {
    return await callCerebrium(endpointUrl, body, FIRST_ATTEMPT_TIMEOUT_MS);
  } catch (firstError) {
    const isTimeout = firstError instanceof CerebriumTimeoutError;
    console.warn(
      `Cerebrium first attempt failed (${isTimeout ? 'timeout' : 'error'}):`,
      firstError instanceof Error ? firstError.message : firstError,
    );
    // Only retry if the first attempt timed out (replica likely still cold).
    // For server-side errors (4xx/5xx) a retry will just hit the same fault.
    if (!isTimeout) throw firstError;
    return await callCerebrium(endpointUrl, body, RETRY_ATTEMPT_TIMEOUT_MS);
  }
}

/** Map a Cerebrium failure to a Next.js response with an accurate status + message. */
function cerebriumErrorResponse(err: unknown): NextResponse {
  if (err instanceof CerebriumTimeoutError) {
    return NextResponse.json(
      {
        error: `Extraction timed out after ${Math.round(err.timeoutMs / 1000)}s. The Cerebrium replica may still be cold-starting — try again in a moment.`,
        kind: 'timeout',
      },
      { status: 504 },
    );
  }
  if (err instanceof CerebriumResponseError) {
    return NextResponse.json(
      {
        error: `Cerebrium responded with ${err.status}: ${err.body.slice(0, 300)}`,
        kind: 'upstream_error',
        status: err.status,
      },
      { status: 502 },
    );
  }
  const msg = err instanceof Error ? err.message : String(err);
  return NextResponse.json(
    { error: `Extraction failed: ${msg}`, kind: 'unknown' },
    { status: 502 },
  );
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text, url, useCanonicalPredicates, useOcr, maxTokens, overlapTokens } = body;

    if (url) {
      return handleUrlExtraction(url, { useOcr, maxTokens, overlapTokens });
    }

    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Missing or invalid text field' }, { status: 400 });
    }

    if (text.length > 10000) {
      return NextResponse.json(
        { error: 'Text too long. Maximum 10,000 characters.' },
        { status: 400 },
      );
    }

    // Wrap text in <page> tags as expected by the model.
    const modelInput = `<page>${text}</page>`;

    // Supabase result cache (separate from Cerebrium's in-memory cache).
    if (CEREBRIUM_EXTRACT_URL && CEREBRIUM_TOKEN) {
      const cachedStatements = await getCachedStatements(text, { useCanonicalPredicates });
      if (cachedStatements) {
        console.log('Returning cached result');
        const result: ExtractionResult = {
          statements: cachedStatements,
          cached: true,
          inputText: text,
        };
        return NextResponse.json(result);
      }
    }

    // Cerebrium (primary production backend). If it errors or times out we
    // surface the actual failure instead of masking with the cached example —
    // that fallback is reserved for "no backend configured".
    if (CEREBRIUM_EXTRACT_URL && CEREBRIUM_TOKEN) {
      try {
        console.log(`Calling Cerebrium /extract: textLen=${text.length}`);
        const payload = (await callCerebriumWithRetry(CEREBRIUM_EXTRACT_URL, {
          text: modelInput,
          useCanonicalPredicates: !!useCanonicalPredicates,
          similarityThreshold: 0.5,
        })) as { statements?: unknown; cached?: boolean } | string;

        const statements = parseStatements(payload as Parameters<typeof parseStatements>[0]);

        // Best-effort persist to Supabase cache (non-fatal).
        if (statements.length > 0) {
          await setCachedStatements(text, statements, { useCanonicalPredicates }).catch(() => {});
        }

        const result: ExtractionResult = {
          statements,
          cached: typeof payload === 'object' && payload !== null && 'cached' in payload
            ? !!(payload as { cached?: boolean }).cached
            : false,
          inputText: text,
        };
        return NextResponse.json(result);
      } catch (cerebriumError) {
        console.warn('Cerebrium call failed:', cerebriumError);
        return cerebriumErrorResponse(cerebriumError);
      }
    }

    // Local model fallback for development.
    if (LOCAL_MODEL_URL) {
      try {
        console.log(`Calling local model: ${LOCAL_MODEL_URL}`);
        const localResponse = await fetch(`${LOCAL_MODEL_URL}/extract`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: modelInput }),
        });

        if (localResponse.ok) {
          const data = await localResponse.json();
          const statements = parseStatements(data.output || data);
          const result: ExtractionResult = {
            statements,
            cached: data.cached || false,
            inputText: text,
          };
          return NextResponse.json(result);
        }
      } catch (localError) {
        console.warn('Local model unavailable:', localError);
      }
    }

    // No backend available — return cached example with a notice.
    console.log('No model endpoint configured, returning cached example');
    return NextResponse.json({
      ...CACHED_EXAMPLE,
      message:
        'No model endpoint configured. Showing cached example. See documentation to run locally or deploy to Cerebrium.',
    });
  } catch (error) {
    console.error('Extract API error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

async function handleUrlExtraction(
  url: string,
  options: { useOcr?: boolean; maxTokens?: number; overlapTokens?: number },
) {
  try {
    new URL(url);
  } catch {
    return NextResponse.json({ error: 'Invalid URL provided' }, { status: 400 });
  }

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return NextResponse.json(
      { error: 'URL must start with http:// or https://' },
      { status: 400 },
    );
  }

  if (CEREBRIUM_EXTRACT_URL_URL && CEREBRIUM_TOKEN) {
    try {
      console.log(`Calling Cerebrium /extract_url: ${url}`);
      const payload = (await callCerebriumWithRetry(CEREBRIUM_EXTRACT_URL_URL, {
        url,
        useOcr: options.useOcr || false,
        maxTokens: options.maxTokens || 1000,
        overlapTokens: options.overlapTokens || 100,
      })) as Record<string, unknown>;

      // The handler returns a DocumentContext model_dump — `statements` is
      // present at the top level alongside metadata/summary, so it parses as
      // a LibraryExtractionResult.
      const statements = parseStatements(payload as unknown as Parameters<typeof parseStatements>[0]);
      return NextResponse.json({
        statements,
        metadata: payload.metadata ?? {
          url,
          chunk_count: 0,
          statement_count: statements.length,
          duplicates_removed: 0,
        },
        summary: payload.summary,
        cached: !!payload.cached,
      });
    } catch (cerebriumError) {
      console.warn('Cerebrium URL extraction failed:', cerebriumError);
      return cerebriumErrorResponse(cerebriumError);
    }
  }

  return NextResponse.json(
    {
      error:
        'URL processing requires Cerebrium. Configure CEREBRIUM_EXTRACT_URL_URL and CEREBRIUM_TOKEN.',
    },
    { status: 503 },
  );
}
