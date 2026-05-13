import { NextRequest, NextResponse } from 'next/server';
import { parseStatements } from '@/lib/statement-parser';
import { getRun } from '@/lib/runs';
import { setCachedStatements } from '@/lib/cache';

// Single Supabase read per call — keep this lightweight, the browser polls
// it every ~2s while a run is in flight.
export const maxDuration = 15;

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  if (!runId) {
    return NextResponse.json({ error: 'Missing runId' }, { status: 400 });
  }

  const row = await getRun(runId);
  if (!row) {
    return NextResponse.json({ error: 'Run not found', status: 'unknown' }, { status: 404 });
  }

  // Still in flight — return the bare status so the client keeps polling.
  if (row.status === 'pending' || row.status === 'running') {
    return NextResponse.json({ run_id: row.run_id, status: row.status });
  }

  if (row.status === 'failed') {
    return NextResponse.json(
      { run_id: row.run_id, status: 'failed', error: row.error ?? 'Unknown error' },
      { status: 200 },
    );
  }

  // status === 'succeeded' — parse the stored result into the same shape the
  // sync route used to return, so page.tsx can render it unchanged.
  const result = row.result as Parameters<typeof parseStatements>[0] | null;
  const statements = result ? parseStatements(result) : [];

  if (row.kind === 'extract_url') {
    const raw = (result as Record<string, unknown> | null) ?? {};
    return NextResponse.json({
      run_id: row.run_id,
      status: 'succeeded',
      statements,
      metadata: raw.metadata ?? {
        url: row.input_text,
        chunk_count: 0,
        statement_count: statements.length,
        duplicates_removed: 0,
      },
      summary: raw.summary,
      cached: !!raw.cached,
    });
  }

  // Text extract: warm the Supabase cache so subsequent identical submissions
  // short-circuit without hitting Cerebrium. Best-effort — failure here
  // must not affect the response.
  if (statements.length > 0 && row.input_text) {
    setCachedStatements(row.input_text, statements).catch(() => {});
  }

  return NextResponse.json({
    run_id: row.run_id,
    status: 'succeeded',
    statements,
    cached: false,
    inputText: row.input_text ?? undefined,
  });
}
