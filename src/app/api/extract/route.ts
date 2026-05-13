import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'node:crypto';
import { parseStatements } from '@/lib/statement-parser';
import { CACHED_EXAMPLE } from '@/lib/cached-example';
import { ExtractionResult } from '@/lib/types';
import { getCachedStatements } from '@/lib/cache';
import { createRun } from '@/lib/runs';

// Submission only — the handler now fires-and-forgets to Cerebrium and
// returns a run_id within a couple of seconds. Long-running pipeline work
// is delivered back to the browser via /api/extract/status/<run_id>.
export const maxDuration = 60;

const CEREBRIUM_EXTRACT_URL = process.env.CEREBRIUM_EXTRACT_URL;
const CEREBRIUM_EXTRACT_URL_URL = process.env.CEREBRIUM_EXTRACT_URL_URL;
const CEREBRIUM_TOKEN = process.env.CEREBRIUM_TOKEN;
const LOCAL_MODEL_URL = process.env.LOCAL_MODEL_URL;

const SUBMIT_TIMEOUT_MS = 30_000;

class CerebriumSubmitError extends Error {
  constructor(public status: number, public body: string) {
    super(`Cerebrium submit failed (${status}): ${body.slice(0, 300)}`);
    this.name = 'CerebriumSubmitError';
  }
}

/** Fire-and-forget POST to a Cerebrium function with ?async=true. Returns
 *  Cerebrium's own run_id (we ignore it; the source of truth for results
 *  is our own run_id which the function uses to write back to Supabase). */
async function submitAsync(endpointUrl: string, body: object): Promise<{ cerebriumRunId: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SUBMIT_TIMEOUT_MS);
  const sep = endpointUrl.includes('?') ? '&' : '?';
  const url = `${endpointUrl}${sep}async=true`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CEREBRIUM_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const errorText = await response.text().catch(() => '<no body>');
      throw new CerebriumSubmitError(response.status, errorText);
    }
    const envelope = (await response.json()) as { run_id?: string };
    return { cerebriumRunId: envelope.run_id ?? '' };
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text, url, useCanonicalPredicates, useOcr, maxTokens, overlapTokens } = body;

    if (url) {
      return handleUrlSubmission(url, { useOcr, maxTokens, overlapTokens });
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

    const modelInput = `<page>${text}</page>`;

    // Supabase result cache — keyed by hash of the user-visible text, so a
    // cache hit short-circuits the async pipeline entirely.
    if (CEREBRIUM_EXTRACT_URL && CEREBRIUM_TOKEN) {
      const cachedStatements = await getCachedStatements(text, { useCanonicalPredicates });
      if (cachedStatements) {
        console.log('Returning cached extraction result');
        const result: ExtractionResult = {
          statements: cachedStatements,
          cached: true,
          inputText: text,
        };
        return NextResponse.json({ ...result, status: 'succeeded' });
      }
    }

    // Primary path: submit async to Cerebrium and return a run_id. The
    // function writes the result back into Supabase keyed by run_id; the
    // browser polls /api/extract/status/<run_id> until it lands.
    if (CEREBRIUM_EXTRACT_URL && CEREBRIUM_TOKEN) {
      const runId = randomUUID();
      try {
        await createRun(runId, 'extract', text);
      } catch (err) {
        console.error('Failed to create run row in Supabase:', err);
        return NextResponse.json(
          { error: `Could not register run: ${err instanceof Error ? err.message : String(err)}` },
          { status: 500 },
        );
      }

      try {
        const { cerebriumRunId } = await submitAsync(CEREBRIUM_EXTRACT_URL, {
          text: modelInput,
          useCanonicalPredicates: !!useCanonicalPredicates,
          similarityThreshold: 0.5,
          run_id: runId,
        });
        console.log(`Submitted extract run=${runId} cerebrium=${cerebriumRunId}`);
        return NextResponse.json({ run_id: runId, status: 'pending' }, { status: 202 });
      } catch (err) {
        console.warn('Cerebrium submit failed:', err);
        return NextResponse.json(
          {
            error: err instanceof CerebriumSubmitError
              ? err.message
              : `Submit failed: ${err instanceof Error ? err.message : String(err)}`,
          },
          { status: 502 },
        );
      }
    }

    // Local-model fallback for development. Still synchronous — local server
    // is fast enough that polling adds no value.
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
          return NextResponse.json({ ...result, status: 'succeeded' });
        }
      } catch (localError) {
        console.warn('Local model unavailable:', localError);
      }
    }

    // No backend at all — keep the demo working with the cached example.
    console.log('No model endpoint configured, returning cached example');
    return NextResponse.json({
      ...CACHED_EXAMPLE,
      status: 'succeeded',
      message:
        'No model endpoint configured. Showing cached example. See documentation to run locally or deploy to Cerebrium.',
    });
  } catch (error) {
    console.error('Extract API error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

async function handleUrlSubmission(
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
  if (!CEREBRIUM_EXTRACT_URL_URL || !CEREBRIUM_TOKEN) {
    return NextResponse.json(
      { error: 'URL processing requires Cerebrium. Configure CEREBRIUM_EXTRACT_URL_URL and CEREBRIUM_TOKEN.' },
      { status: 503 },
    );
  }

  const runId = randomUUID();
  try {
    await createRun(runId, 'extract_url', url);
  } catch (err) {
    console.error('Failed to create URL run row:', err);
    return NextResponse.json(
      { error: `Could not register run: ${err instanceof Error ? err.message : String(err)}` },
      { status: 500 },
    );
  }

  try {
    const { cerebriumRunId } = await submitAsync(CEREBRIUM_EXTRACT_URL_URL, {
      url,
      useOcr: options.useOcr || false,
      maxTokens: options.maxTokens || 1000,
      overlapTokens: options.overlapTokens || 100,
      run_id: runId,
    });
    console.log(`Submitted extract_url run=${runId} cerebrium=${cerebriumRunId} url=${url}`);
    return NextResponse.json({ run_id: runId, status: 'pending' }, { status: 202 });
  } catch (err) {
    console.warn('Cerebrium URL submit failed:', err);
    return NextResponse.json(
      {
        error: err instanceof CerebriumSubmitError
          ? err.message
          : `URL submit failed: ${err instanceof Error ? err.message : String(err)}`,
      },
      { status: 502 },
    );
  }
}
