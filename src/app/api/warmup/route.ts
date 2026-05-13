import { NextResponse } from 'next/server';

/**
 * Fire-and-forget warmup ping.
 *
 * The browser calls this on page load to wake the Cerebrium replica before
 * the user submits a real request. We submit the extract function with
 * `?async=true` and NO webhookEndpoint — Cerebrium spins up the replica,
 * runs the function (it does a tiny `warmup` payload), and forwards the
 * result nowhere. No Supabase row, no poll, no orphan record. The replica
 * stays warm for `cooldown` seconds afterwards.
 */
export const maxDuration = 10;

const CEREBRIUM_EXTRACT_URL = process.env.CEREBRIUM_EXTRACT_URL;
const CEREBRIUM_TOKEN = process.env.CEREBRIUM_TOKEN;

export async function POST() {
  if (!CEREBRIUM_EXTRACT_URL || !CEREBRIUM_TOKEN) {
    // Nothing to warm — local dev or unconfigured deploy. Return OK silently
    // so the browser useEffect doesn't surface a useless error.
    return NextResponse.json({ ok: true, skipped: 'no-backend' });
  }

  const url = `${CEREBRIUM_EXTRACT_URL}${CEREBRIUM_EXTRACT_URL.includes('?') ? '&' : '?'}async=true`;

  // 5s budget — Cerebrium typically 202s in well under a second. We don't
  // care about the response body; just kicking the replica.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5_000);
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CEREBRIUM_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: '<page>warmup</page>' }),
      signal: controller.signal,
    });
    return NextResponse.json({ ok: resp.ok, status: resp.status });
  } catch (err) {
    // Network/timeout errors are fine here — the user-facing flow isn't
    // blocked on the warmup. Log and shrug.
    console.warn('warmup ping failed:', err);
    return NextResponse.json({ ok: false, error: err instanceof Error ? err.message : 'unknown' });
  } finally {
    clearTimeout(timer);
  }
}
