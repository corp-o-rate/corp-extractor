import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

const TOPICS = [
  'a technology company announcing a new product or partnership',
  'a government policy change affecting businesses or citizens',
  'a scientific discovery with real-world applications',
  'an environmental initiative by a major corporation',
  'a merger or acquisition between two companies',
  'a sports team or athlete achieving a milestone',
  'a cultural event or festival and its economic impact',
  'a healthcare breakthrough or public health announcement',
  'an infrastructure project in a major city',
  'a financial institution reporting quarterly results',
  'an education reform or university research finding',
  'a legal case with implications for an industry',
];

export async function POST() {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return new Response(
      JSON.stringify({ error: 'OpenAI API key not configured' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } }
    );
  }

  const topic = TOPICS[Math.floor(Math.random() * TOPICS.length)];

  try {
    const result = streamText({
      model: openai('gpt-4o-mini'),
      prompt: `Write a realistic news article or press release about ${topic}.

Requirements:
- Write exactly 2 paragraphs
- Maximum 1000 words total
- Use ONLY real, verifiable people, organizations, and places — no fictional or invented names.
- Every person named must be a real public figure, and the role attributed to them (CEO, founder, minister, professor, etc.) must be their actual, current (or last-known) role at the named organization. Do not invent roles or assign people to organizations they don't actually belong to.
- Every company, agency, university, or other organization named must be a real entity that exists today.
- Cities, countries, products, and other proper nouns must also be real.
- Include dates, numbers, and statistics where appropriate; these should be plausible but do not need to be exact.
- Write in a professional journalistic style with clear subject-predicate-object relationships.
- Do not include a headline or title, just the body text.`,
    });

    return result.toTextStreamResponse();
  } catch (error) {
    console.error('Text generation error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to generate text' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
