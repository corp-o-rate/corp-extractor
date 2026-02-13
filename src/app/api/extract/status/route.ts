import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { setCachedStatements } from '@/lib/cache';

const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

export async function GET(request: NextRequest) {
  const jobId = request.nextUrl.searchParams.get('jobId');
  const inputText = request.nextUrl.searchParams.get('inputText');
  const useCanonicalPredicates = request.nextUrl.searchParams.get('useCanonicalPredicates') === 'true';

  if (!jobId) {
    return NextResponse.json(
      { error: 'Missing jobId parameter' },
      { status: 400 }
    );
  }

  if (!RUNPOD_ENDPOINT_ID || !RUNPOD_API_KEY) {
    return NextResponse.json(
      { error: 'RunPod not configured' },
      { status: 500 }
    );
  }

  try {
    console.log(`Checking status for job: ${jobId}`);

    const response = await fetch(
      `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${RUNPOD_API_KEY}`,
        },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`RunPod status error: status=${response.status}, body=${errorText}`);
      return NextResponse.json(
        { error: `Failed to check status: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log(`Job ${jobId} status: ${data.status}`);

    // Handle completed job - parse the output
    if (data.status === 'COMPLETED' && data.output) {
      // Check if this is a URL/document job result (DocumentContext model_dump format)
      if (data.output.document && data.output.labeled_statements) {
        // DocumentContext model_dump format — use parseStatements which handles model_dump
        const statements = parseStatements(data.output);

        return NextResponse.json({
          status: 'COMPLETED',
          statements,
          metadata: {
            title: data.output.document?.metadata?.title,
            chunk_count: data.output.chunks?.length || 0,
            statement_count: data.output.labeled_statements?.length || 0,
            duplicates_removed: (data.output.pre_dedup_count || 0) - (data.output.post_dedup_count || 0),
          },
          summary: data.output.document?.summary,
          cached: data.output.cached || false,
        });
      }

      // Regular text extraction result (ExtractionResult model_dump format)
      // Also handle legacy format with output.output wrapper
      const outputData = data.output.output || data.output;
      const statements = parseStatements(outputData);

      // Cache the result if we have the input text
      // Note: setCachedStatements will skip empty results to prevent caching failures/timeouts
      if (inputText && statements.length > 0) {
        await setCachedStatements(inputText, statements, { useCanonicalPredicates });
      }

      return NextResponse.json({
        status: 'COMPLETED',
        statements,
        cached: data.output.cached || false,
      });
    }

    // Handle failed job
    if (data.status === 'FAILED') {
      console.error(`Job ${jobId} failed:`, data.error);
      // DO NOT cache failed results
      return NextResponse.json({
        status: 'FAILED',
        error: data.error || 'Job failed',
      });
    }

    // Handle timed out job - DO NOT cache
    if (data.status === 'TIMED_OUT') {
      console.error(`Job ${jobId} timed out`);
      return NextResponse.json({
        status: 'TIMED_OUT',
        error: 'Request timed out. The server may be busy or starting up. Please try again.',
      });
    }

    // Job still in progress
    return NextResponse.json({
      status: data.status, // IN_QUEUE or IN_PROGRESS
    });

  } catch (error) {
    console.error('Status check error:', error);
    return NextResponse.json(
      { error: 'Failed to check job status' },
      { status: 500 }
    );
  }
}
