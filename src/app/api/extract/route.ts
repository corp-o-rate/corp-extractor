import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { CACHED_EXAMPLE } from '@/lib/cached-example';
import { ExtractionResult } from '@/lib/types';
import { getCachedStatements, setCachedStatements } from '@/lib/cache';

// Cold-boot of a Cerebrium replica (T5-Gemma2 + GLiNER2 + GGUF qualifier
// loading from /persistent-storage) can take several minutes on the very
// first hit after scale-to-zero. Vercel hobby caps function duration at
// 300s; we use the full window and rely on a one-shot retry below to land
// the second attempt on a now-warm replica.
export const maxDuration = 300;

const CEREBRIUM_EXTRACT_URL = process.env.CEREBRIUM_EXTRACT_URL;
const CEREBRIUM_EXTRACT_URL_URL = process.env.CEREBRIUM_EXTRACT_URL_URL;
const CEREBRIUM_TOKEN = process.env.CEREBRIUM_TOKEN;
const LOCAL_MODEL_URL = process.env.LOCAL_MODEL_URL;

// Per-attempt client-side timeout. We give the first attempt 240s so a
// retry has ~60s to land if the first aborts. If neither attempt finishes
// within the 300s function budget, Vercel kills the request and the
// frontend's warmup ping (page.tsx) will have spun the replica up by the
// next user action.
const FIRST_ATTEMPT_TIMEOUT_MS = 240_000;
const RETRY_ATTEMPT_TIMEOUT_MS = 50_000;

interface CerebriumEnvelope {
  run_id?: string;
  result?: unknown;
  run_time_ms?: number;
}

async function callCerebrium(endpointUrl: string, body: object, timeoutMs: number): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(endpointUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CEREBRIUM_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Cerebrium error: status=${response.status}, body=${errorText.slice(0, 500)}`);
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
    const isAbort = firstError instanceof Error && firstError.name === 'AbortError';
    console.warn(
      `Cerebrium first attempt failed (${isAbort ? 'timeout' : 'error'}):`,
      firstError instanceof Error ? firstError.message : firstError,
    );
    // Retry once. The cold-start should have triggered the replica to spin
    // up during the first attempt, so the second hit usually returns fast.
    return await callCerebrium(endpointUrl, body, RETRY_ATTEMPT_TIMEOUT_MS);
  }
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

    // Cerebrium (primary production backend).
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
        console.warn('Cerebrium unavailable:', cerebriumError);
        // Fall through to local model / cached example.
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
      console.warn('Cerebrium unavailable for URL processing:', cerebriumError);
      return NextResponse.json(
        {
          error:
            cerebriumError instanceof Error
              ? cerebriumError.message
              : 'Cerebrium error during URL processing',
        },
        { status: 502 },
      );
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
