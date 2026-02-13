import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { CACHED_EXAMPLE } from '@/lib/cached-example';
import { ExtractionResult, UrlExtractionResult } from '@/lib/types';
import { getCachedStatements } from '@/lib/cache';

// Environment configuration
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const LOCAL_MODEL_URL = process.env.LOCAL_MODEL_URL;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text, url, useCanonicalPredicates, useOcr, maxTokens, overlapTokens } = body;

    // Handle URL input
    if (url) {
      return handleUrlExtraction(url, { useOcr, maxTokens, overlapTokens });
    }

    if (!text || typeof text !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid text field' },
        { status: 400 }
      );
    }

    if (text.length > 10000) {
      return NextResponse.json(
        { error: 'Text too long. Maximum 10,000 characters.' },
        { status: 400 }
      );
    }

    // Wrap text in page tags as expected by the model
    const modelInput = `<page>${text}</page>`;

    // Check cache first (for RunPod requests)
    if (RUNPOD_ENDPOINT_ID && RUNPOD_API_KEY) {
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

    // Try RunPod first (primary production option) - use async /run endpoint
    if (RUNPOD_ENDPOINT_ID && RUNPOD_API_KEY) {
      try {
        console.log(`Submitting job to RunPod endpoint: ${RUNPOD_ENDPOINT_ID}`);

        const runpodResponse = await fetch(
          `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${RUNPOD_API_KEY}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              input: {
                text: modelInput,
                format: 'json', // Request JSON format for v0.2.0+ (includes confidence scores)
                useCanonicalPredicates: useCanonicalPredicates || false,
              },
            }),
          }
        );

        if (runpodResponse.ok) {
          const data = await runpodResponse.json();
          console.log(`RunPod job submitted: ${data.id}, status: ${data.status}`);

          // Return job ID for polling
          return NextResponse.json({
            jobId: data.id,
            status: data.status,
            inputText: text,
          });
        } else {
          const errorText = await runpodResponse.text();
          console.error(`RunPod API error: status=${runpodResponse.status}, body=${errorText}`);
          throw new Error(`RunPod API error: ${runpodResponse.status}`);
        }
      } catch (runpodError) {
        console.warn('RunPod unavailable:', runpodError);
        // Fall through to next option
      }
    }

    // Try local model if configured (returns result directly, no polling needed)
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
          // New format: ExtractionResult model_dump {statements: [...], source_text, cached}
          // Also support legacy format: {output: "xml string", success: true}
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

    // No model available, return cached example
    console.log('No model endpoint configured, returning cached example');
    return NextResponse.json({
      ...CACHED_EXAMPLE,
      message: 'No model endpoint configured. Showing cached example. See documentation to run locally or deploy to RunPod.',
    });

  } catch (error) {
    console.error('Extract API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

async function handleUrlExtraction(
  url: string,
  options: { useOcr?: boolean; maxTokens?: number; overlapTokens?: number }
) {
  // Validate URL
  try {
    new URL(url);
  } catch {
    return NextResponse.json(
      { error: 'Invalid URL provided' },
      { status: 400 }
    );
  }

  // Only allow http/https
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return NextResponse.json(
      { error: 'URL must start with http:// or https://' },
      { status: 400 }
    );
  }

  // Try RunPod first (primary production option) - use async /run endpoint
  if (RUNPOD_ENDPOINT_ID && RUNPOD_API_KEY) {
    try {
      console.log(`Submitting URL job to RunPod endpoint: ${RUNPOD_ENDPOINT_ID}`);

      const runpodResponse = await fetch(
        `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${RUNPOD_API_KEY}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            input: {
              url,
              useOcr: options.useOcr || false,
              maxTokens: options.maxTokens || 1000,
              overlapTokens: options.overlapTokens || 100,
            },
          }),
        }
      );

      if (runpodResponse.ok) {
        const data = await runpodResponse.json();
        console.log(`RunPod URL job submitted: ${data.id}, status: ${data.status}`);

        // Return job ID for polling
        return NextResponse.json({
          jobId: data.id,
          status: data.status,
          inputUrl: url,
          isUrlJob: true,
        });
      } else {
        const errorText = await runpodResponse.text();
        console.error(`RunPod API error: status=${runpodResponse.status}, body=${errorText}`);
        throw new Error(`RunPod API error: ${runpodResponse.status}`);
      }
    } catch (runpodError) {
      console.warn('RunPod unavailable for URL processing:', runpodError);
      // Fall through to error
    }
  }

  // No model available for URL processing
  return NextResponse.json(
    { error: 'URL processing requires RunPod endpoint. Configure RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY.' },
    { status: 503 }
  );
}
