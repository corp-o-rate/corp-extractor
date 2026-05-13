import { NextRequest, NextResponse } from 'next/server';
import { updateRun, verifyWebhookToken } from '@/lib/runs';

// Cerebrium's webhook forwarder POSTs the function's return value here when
// the run completes. The URL is unguessable — `runId` is a UUID and we
// require a matching HMAC token derived from a Vercel-only secret — so this
// route doesn't need any other auth.
export const maxDuration = 30;

interface CerebriumWebhookPayload {
  result?: unknown;
  run_id?: string;
  error?: string;
  // The function's actual return value gets nested under `result` by
  // Cerebrium's forwarder; we also accept top-level fields as a fallback in
  // case the envelope shape changes.
  [key: string]: unknown;
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  const token = new URL(request.url).searchParams.get('token') ?? '';
  if (!verifyWebhookToken(runId, token)) {
    console.warn(`webhook ${runId}: invalid or missing token`);
    return NextResponse.json({ error: 'invalid token' }, { status: 401 });
  }

  let payload: CerebriumWebhookPayload;
  try {
    payload = (await request.json()) as CerebriumWebhookPayload;
  } catch (e) {
    console.warn(`webhook ${runId}: invalid JSON body`, e);
    return NextResponse.json({ error: 'invalid json' }, { status: 400 });
  }

  // The function-return value is either at `.result` (Cerebrium's
  // standard envelope) or at the top level (some webhook configs flatten).
  const inner = (payload.result ?? payload) as Record<string, unknown>;
  const error = (inner.error as string | undefined) ?? (payload.error as string | undefined);

  try {
    if (error) {
      await updateRun(runId, { status: 'failed', error, completed: true });
    } else {
      await updateRun(runId, { status: 'succeeded', result: inner, completed: true });
    }
  } catch (err) {
    console.error(`webhook ${runId}: updateRun failed:`, err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'updateRun failed' },
      { status: 500 },
    );
  }

  return NextResponse.json({ ok: true });
}
