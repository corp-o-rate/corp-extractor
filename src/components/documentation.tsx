'use client';

import { useState, useEffect } from 'react';
import { Copy, Check, ExternalLink, Terminal, Code2, Server, Cloud } from 'lucide-react';
import { toast } from 'sonner';

type TabId = 'cli' | 'python' | 'typescript' | 'runpod' | 'local' | 'output';

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'cli', label: 'CLI', icon: <Terminal className="w-4 h-4" /> },
  { id: 'python', label: 'Python', icon: <Code2 className="w-4 h-4" /> },
  { id: 'typescript', label: 'TypeScript', icon: <Code2 className="w-4 h-4" /> },
  { id: 'runpod', label: 'RunPod', icon: <Cloud className="w-4 h-4" /> },
  { id: 'local', label: 'Run Locally', icon: <Server className="w-4 h-4" /> },
  { id: 'output', label: 'Output Format', icon: <Terminal className="w-4 h-4" /> },
];

const HF_MODEL = 'Corp-o-Rate-Community/statement-extractor';
const PYPI_PACKAGE = 'corp-extractor';

const CODE_SNIPPETS: Record<TabId, string> = {
  cli: `# Command Line Interface (v0.2.4+)

# ============================================
# Install globally (recommended)
# ============================================

# Using uv (recommended)
uv tool install "${PYPI_PACKAGE}[embeddings]"

# Or using pipx
pipx install "${PYPI_PACKAGE}[embeddings]"

# Or using pip
pip install "${PYPI_PACKAGE}[embeddings]"

# ============================================
# Quick run with uvx (no install)
# ============================================
# Note: First run downloads the model (~1.5GB)
uvx ${PYPI_PACKAGE} "Apple announced a new iPhone."

# ============================================
# Usage Examples
# ============================================

# Extract from text argument
corp-extractor "Apple Inc. announced the iPhone 15 at their September event."

# Extract from file
corp-extractor -f article.txt

# Pipe from stdin
cat article.txt | corp-extractor -

# Output as JSON (with full metadata)
corp-extractor "Tim Cook is CEO of Apple." --json

# Output as XML (raw model output)
corp-extractor -f article.txt --xml

# Verbose output with confidence scores
corp-extractor -f article.txt --verbose

# Use more beams for better quality
corp-extractor -f article.txt --beams 8

# Use custom predicate taxonomy
corp-extractor -f article.txt --taxonomy predicates.txt

# Use GPU explicitly
corp-extractor -f article.txt --device cuda

# Filter low-confidence results
corp-extractor -f article.txt --min-confidence 0.7

# ============================================
# Persistent Server (v0.9.7)
# ============================================

# Start a server to keep models warm (avoids ~30s startup)
corp-extractor serve                    # Start on localhost:8111
corp-extractor serve --port 9000        # Custom port

# Use --server to delegate to a running server
corp-extractor --server pipeline "Amazon CEO Andy Jassy..."
corp-extractor --server split -f article.txt --json

# Or set the environment variable
export CORP_EXTRACTOR_SERVER=http://localhost:8111
corp-extractor pipeline "text"  # Automatically uses the server

# ============================================
# All CLI Options
# ============================================
# corp-extractor --help
#
# -f, --file PATH              Read input from file
# -o, --output [table|json|xml] Output format (default: table)
# --json                       Output as JSON (shortcut)
# --xml                        Output as XML (shortcut)
# -b, --beams INTEGER          Number of beams (default: 4)
# --diversity FLOAT            Diversity penalty (default: 1.0)
# --max-tokens INTEGER         Max tokens to generate (default: 2048)
# --no-dedup                   Disable deduplication
# --no-embeddings              Disable embedding-based dedup (faster)
# --no-merge                   Disable beam merging
# --predicates PATH            Load predicate list for GLiNER2 relation extraction
# --all-triples                Keep all candidate triples (default: best per source)
# --dedup-threshold FLOAT      Deduplication threshold (default: 0.65)
# --min-confidence FLOAT       Min confidence filter (default: 0)
# --taxonomy PATH              Load predicate taxonomy from file
# --taxonomy-threshold FLOAT   Taxonomy matching threshold (default: 0.5)
# --device [auto|cuda|mps|cpu] Device to use (default: auto)
# -v, --verbose                Show confidence scores and metadata
# -q, --quiet                  Suppress progress messages
# --version                    Show version`,

  python: `# Installation
pip install "${PYPI_PACKAGE}"

# GLiNER2 model downloads automatically on first use (~800MB)

# For GPU support, install PyTorch with CUDA first:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# ============================================
# Simple Usage - Returns Pydantic Models
# ============================================
from statement_extractor import extract_statements

text = """
Apple Inc. announced a commitment to carbon neutrality by 2030.
Tim Cook presented the sustainability report to investors.
"""

result = extract_statements(text)

for stmt in result:
    print(f"{stmt.subject.text} ({stmt.subject.type})")
    print(f"  --[{stmt.predicate}]--> {stmt.object.text}")
    print(f"  Confidence: {stmt.confidence_score:.2f}")
    print()

# ============================================
# NEW in v0.4.0: GLiNER2 Integration
# ============================================
# The library uses GLiNER2 (205M params) for:
# - Entity recognition and boundary refinement
# - Relation extraction when predicates are provided
# - Entity-based confidence scoring

# Use GLiNER2 relation extraction with predefined predicates:
from statement_extractor import ExtractionOptions
options = ExtractionOptions(predicates=["works_for", "founded", "acquired"])
result = extract_statements(text, options)

# ============================================
# Quality Scoring & Beam Merging (v0.2.0+)
# ============================================
# Combined scoring: 50% semantic similarity + 25% subject/object noun scores

from statement_extractor import ExtractionOptions, ScoringConfig

# Precision mode - filter low-confidence triples
scoring = ScoringConfig(min_confidence=0.7)
options = ExtractionOptions(scoring_config=scoring)
result = extract_statements(text, options)

# ============================================
# NEW in v0.2.0: Predicate Taxonomy
# ============================================
from statement_extractor import PredicateTaxonomy

# Normalize predicates to canonical forms using embeddings
taxonomy = PredicateTaxonomy(predicates=[
    "acquired", "founded", "works_for", "announced",
    "invested_in", "partnered_with", "committed_to"
])

options = ExtractionOptions(predicate_taxonomy=taxonomy)
result = extract_statements(text, options)

# "committed to" matches "committed_to" via embedding similarity
for stmt in result:
    if stmt.canonical_predicate:
        print(f"Normalized: {stmt.predicate} -> {stmt.canonical_predicate}")

# ============================================
# Disable embeddings (faster, no extra deps)
# ============================================
options = ExtractionOptions(
    embedding_dedup=False,  # Use exact text matching
    merge_beams=False,      # Select single best beam
)
result = extract_statements(text, options)

# ============================================
# Alternative Output Formats
# ============================================
from statement_extractor import (
    extract_statements_as_json,
    extract_statements_as_xml,
    extract_statements_as_dict,
)

json_output = extract_statements_as_json(text)
xml_output = extract_statements_as_xml(text)
dict_output = extract_statements_as_dict(text)

# ============================================
# Server Delegation (v0.9.8)
# ============================================
# Delegate to a running server — no local GPU needed
result = extract_statements(text, server_url="http://localhost:8111")

# Pipeline and document pipeline also support server_url
from statement_extractor.pipeline import ExtractionPipeline
pipeline = ExtractionPipeline(server_url="http://localhost:8111")
ctx = pipeline.process(text)

# ============================================
# Batch Processing
# ============================================
from statement_extractor import StatementExtractor

extractor = StatementExtractor(device="cuda")  # or "cpu"

texts = ["Text 1...", "Text 2...", "Text 3..."]
for text in texts:
    result = extractor.extract(text)
    print(f"Found {len(result)} statements")

# Library uses Diverse Beam Search: https://arxiv.org/abs/1610.02424`,

  typescript: `// Installation
// npm install @huggingface/inference

import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HF_TOKEN);

async function extractStatements(text: string): Promise<string> {
  // Note: For seq2seq models, you may need to use a custom endpoint
  // or run locally since HF Inference API has limited seq2seq support
  const response = await hf.textGeneration({
    model: "${HF_MODEL}",
    inputs: \`<page>\${text}</page>\`,
    parameters: {
      max_new_tokens: 2048,
      num_beams: 4,
    },
  });

  return response.generated_text;
}

// For production use, we recommend RunPod or running locally
// See the "RunPod" or "Run Locally" tabs for setup instructions

// Example usage
const text = "Apple Inc. announced a commitment to carbon neutrality by 2030.";
const result = await extractStatements(text);
console.log(result);`,

  runpod: `# Deploy to RunPod Serverless (~$0.0002/sec GPU)

## 1. Build and push Docker image
\`\`\`bash
git clone https://github.com/corp-o-rate/statement-extractor
cd statement-extractor/runpod

# Build the image (--platform flag required on Mac)
docker build --platform linux/amd64 -t statement-extractor-runpod .

# Push to Docker Hub
docker tag statement-extractor-runpod YOUR_USERNAME/statement-extractor-runpod
docker push YOUR_USERNAME/statement-extractor-runpod
\`\`\`

## 2. Create RunPod Endpoint
1. Go to runpod.io/console/serverless?ref=sjoylkgj
2. Click "New Endpoint"
3. Set container image to your pushed image
4. Select GPU (RTX 3090+ recommended)
5. Set Active Workers: 0, Max Workers: 1-3

## 3. Call the API
\`\`\`bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"input": {"text": "<page>Your text here</page>"}}'
\`\`\`

## TypeScript Client
\`\`\`typescript
const response = await fetch(
  \`https://api.runpod.ai/v2/\${ENDPOINT_ID}/runsync\`,
  {
    method: 'POST',
    headers: {
      'Authorization': \`Bearer \${RUNPOD_API_KEY}\`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: { text: \`<page>\${text}</page>\` },
    }),
  }
);
const data = await response.json();
const output = data.output.output; // XML statements
\`\`\`

## Pricing (pay only when processing)
- RTX 3090: ~$0.00031/sec (~$1.12/hr active)
- Idle: $0 (scales to zero)
- 1000 requests/day: ~$1.86/month`,

  local: `# Run Locally (No API Limits)

## 1. Clone the demo site
\`\`\`bash
git clone https://github.com/corp-o-rate/statement-extractor
cd statement-extractor
pnpm install
\`\`\`

## 2. Install uv and download the model
\`\`\`bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download the model
uv run huggingface-cli download ${HF_MODEL} --local-dir ./model
\`\`\`

## 3. Run the local API server
\`\`\`bash
cd local-server
cp .env.example .env  # Edit .env to set MODEL_PATH
uv sync
uv run python server.py
\`\`\`

## 4. Start the frontend
\`\`\`bash
# Point to local API
echo "LOCAL_MODEL_URL=http://localhost:8000" >> .env.local
pnpm dev
\`\`\`

## Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only (slow, ~30s per extraction)
- **Recommended**: 16GB RAM + CUDA GPU (fast, ~2s per extraction)
- **Model size**: ~1.5GB disk space`,

  output: `<!-- Output Format -->
<!-- The model outputs XML with extracted statements -->

<statements>
  <stmt>
    <subject type="ORG">Apple Inc.</subject>
    <object type="EVENT">carbon neutrality by 2030</object>
    <predicate>committed to</predicate>
    <text>Apple Inc. committed to achieving carbon neutrality by 2030.</text>
  </stmt>
  <stmt>
    <subject type="PERSON">Tim Cook</subject>
    <object type="MONEY">$4.7 billion</object>
    <predicate>announced investment of</predicate>
    <text>Tim Cook announced an investment of $4.7 billion.</text>
  </stmt>
</statements>

<!-- Entity Types -->
ORG       - Organizations (companies, agencies)
PERSON    - People (names, titles)
GPE       - Geopolitical entities (countries, cities)
LOC       - Locations (mountains, rivers)
PRODUCT   - Products (devices, services)
EVENT     - Events (announcements, meetings)
WORK_OF_ART - Creative works (reports, books)
LAW       - Legal documents
DATE      - Dates and time periods
MONEY     - Monetary values
PERCENT   - Percentages
QUANTITY  - Quantities and measurements`,
};

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success('Copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="copy-btn"
      title="Copy to clipboard"
    >
      {copied ? (
        <>
          <Check className="w-4 h-4 text-green-500" />
          Copied!
        </>
      ) : (
        <>
          <Copy className="w-4 h-4" />
          Copy
        </>
      )}
    </button>
  );
}

// Map tab IDs to languages for syntax highlighting
const TAB_LANGUAGES: Record<TabId, string> = {
  cli: 'bash',
  python: 'python',
  typescript: 'typescript',
  runpod: 'bash',
  local: 'bash',
  output: 'xml',
};

function HighlightedCode({ code, language }: { code: string; language: string }) {
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null);

  useEffect(() => {
    const highlight = async () => {
      try {
        const { highlightCode } = await import('@/lib/shiki');
        const html = await highlightCode(code, language as any);
        setHighlightedHtml(html);
      } catch (error) {
        console.error('Failed to highlight code:', error);
        setHighlightedHtml(null);
      }
    };
    highlight();
  }, [code, language]);

  if (highlightedHtml) {
    return (
      <div
        className="shiki-wrapper [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!m-0 [&_code]:!bg-transparent"
        dangerouslySetInnerHTML={{ __html: highlightedHtml }}
      />
    );
  }

  return (
    <pre>
      <code>{code}</code>
    </pre>
  );
}

export function QuickStart() {
  const [activeTab, setActiveTab] = useState<TabId>('cli');

  return (
    <div id="quick-start">
      {/* Tab list */}
      <div className="tab-list overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className="tab-trigger flex items-center gap-2 whitespace-nowrap"
            data-state={activeTab === tab.id ? 'active' : 'inactive'}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="mt-4">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="flex items-center gap-4 flex-wrap">
            {(activeTab === 'cli' || activeTab === 'python') && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">PyPI:</span>
                <a
                  href={`https://pypi.org/project/${PYPI_PACKAGE}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-red-600 hover:underline flex items-center gap-1"
                >
                  {PYPI_PACKAGE}
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            )}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-500">Model:</span>
              <a
                href={`https://huggingface.co/${HF_MODEL}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-red-600 hover:underline flex items-center gap-1"
              >
                {HF_MODEL}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
          <CopyButton text={CODE_SNIPPETS[activeTab]} />
        </div>

        <div className="code-block overflow-x-auto max-h-[500px] overflow-y-auto">
          <HighlightedCode
            code={CODE_SNIPPETS[activeTab]}
            language={TAB_LANGUAGES[activeTab]}
          />
        </div>

        <div className="mt-6 text-center">
          <a
            href="/docs"
            className="text-red-600 hover:underline font-medium"
          >
            View full documentation &rarr;
          </a>
        </div>
      </div>
    </div>
  );
}

// Keep old export for backwards compatibility
export const Documentation = QuickStart;
