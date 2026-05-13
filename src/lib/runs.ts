/**
 * Async run tracking. Vercel inserts a 'pending' row when submitting work
 * to Cerebrium; the Cerebrium handler writes the result back to the same
 * row when it finishes (see cerebrium/main.py). The browser polls
 * /api/extract/status/<run_id> which calls getRun() here.
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

export type RunStatus = 'pending' | 'running' | 'succeeded' | 'failed';
export type RunKind = 'extract' | 'extract_url';

export interface RunRow {
  run_id: string;
  status: RunStatus;
  result: unknown | null;
  error: string | null;
  kind: RunKind;
  input_text: string | null;
  created_at: string;
  completed_at: string | null;
}

function getClient(): SupabaseClient | null {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) return null;
  return createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
}

/** Insert a freshly-submitted run. Throws on Supabase error — callers should treat
 *  Supabase as a hard dependency for the async flow (no point submitting work we
 *  can't retrieve). */
export async function createRun(
  runId: string,
  kind: RunKind,
  inputText: string | null,
): Promise<void> {
  const supabase = getClient();
  if (!supabase) throw new Error('Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing)');
  const { error } = await supabase.from('extraction_runs').insert({
    run_id: runId,
    status: 'pending',
    kind,
    input_text: inputText,
  });
  if (error) throw new Error(`Failed to create run row: ${error.message}`);
}

export async function getRun(runId: string): Promise<RunRow | null> {
  const supabase = getClient();
  if (!supabase) return null;
  const { data, error } = await supabase
    .from('extraction_runs')
    .select('*')
    .eq('run_id', runId)
    .maybeSingle();
  if (error) {
    console.warn('getRun error:', error);
    return null;
  }
  return data as RunRow | null;
}
