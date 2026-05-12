/**
 * Core types for statement extraction
 */

export type EntityType =
  | 'ORG'
  | 'PERSON'
  | 'GPE'
  | 'LOC'
  | 'PRODUCT'
  | 'EVENT'
  | 'WORK_OF_ART'
  | 'LAW'
  | 'DATE'
  | 'MONEY'
  | 'PERCENT'
  | 'QUANTITY'
  | 'UNKNOWN';

export type ExtractionMethod = 'hybrid' | 'spacy' | 'split' | 'model' | 'gliner_relation';

/** Flattened qualifier fields surfaced from EntityQualifiers + identifiers. */
export interface EntityQualifiers {
  legal_name?: string;
  org?: string;
  role?: string;
  region?: string;
  country?: string;
  city?: string;
  jurisdiction?: string;
  source?: string;
  source_id?: string;
  /** Remaining identifiers (lei, ch_number, sec_cik, ticker, wikidata_qid, ...) */
  identifiers?: Record<string, string>;
}

export interface Entity {
  /** Canonical display name when resolved, else the raw extracted text. */
  name: string;
  type: EntityType;
  /** The raw extracted span from the source text. */
  text?: string;
  /** Fully qualified name, e.g. "Tim Cook (CEO, Apple Inc)". */
  fqn?: string;
  /** Canonical ID (e.g. "WIKIDATA:Q123", "LEI:5493001KJTIIGC8Y1R12"). */
  canonicalId?: string;
  /** Qualifier fields flattened for display. */
  qualifiers?: EntityQualifiers;
  /** Canonical-match metadata, when available. */
  matchMethod?: string;
  matchConfidence?: number;
}

/** Label applied to a statement (sentiment, relation_type, etc.) */
export interface StatementLabel {
  label_type: string;
  label_value: string | number | boolean;
  confidence: number;
  labeler?: string;
}

/** Taxonomy classification result */
export interface TaxonomyResult {
  taxonomy_name: string;
  category: string;
  label: string;
  label_id?: number;
  confidence: number;
  classifier?: string;
}

export interface Statement {
  subject: Entity;
  object: Entity;
  predicate: string;
  /** Category of the predicate (e.g., 'ownership_control', 'employment_leadership') */
  predicateCategory?: string;
  text: string;
  /** Semantic similarity score (0-1) between source text and reassembled triple */
  confidence?: number;
  /** Canonical form of the predicate if taxonomy matching was used */
  canonicalPredicate?: string;
  /** Method used to extract this triple (hybrid, spacy, split, or model) */
  extractionMethod?: ExtractionMethod;
  /** Labels applied to this statement (sentiment, relation_type, etc.) */
  labels?: StatementLabel[];
  /** Taxonomy classification results */
  taxonomyResults?: TaxonomyResult[];
}

export interface ExtractionResult {
  statements: Statement[];
  cached: boolean;
  message?: string;
  inputText?: string;
}

// URL extraction result with metadata (returned by Cerebrium /extract_url).
export interface UrlExtractionResult {
  statements: Statement[];
  metadata: {
    title?: string;
    url: string;
    chunk_count: number;
    statement_count: number;
    duplicates_removed: number;
  };
  summary?: string;
  cached?: boolean;
}

export interface GraphNode {
  id: string;
  name: string;
  type: EntityType;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  predicate: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

/**
 * Get CSS class for entity type badge
 */
export function getEntityBadgeClass(type: EntityType): string {
  const typeMap: Record<EntityType, string> = {
    ORG: 'badge-org',
    PERSON: 'badge-person',
    GPE: 'badge-gpe',
    LOC: 'badge-loc',
    PRODUCT: 'badge-product',
    EVENT: 'badge-event',
    WORK_OF_ART: 'badge-default',
    LAW: 'badge-default',
    DATE: 'badge-default',
    MONEY: 'badge-default',
    PERCENT: 'badge-default',
    QUANTITY: 'badge-default',
    UNKNOWN: 'badge-default',
  };
  return typeMap[type] || 'badge-default';
}

/**
 * Get color for entity type (for graph visualization)
 */
export function getEntityColor(type: EntityType): string {
  const colorMap: Record<EntityType, string> = {
    ORG: '#3b82f6',
    PERSON: '#8b5cf6',
    GPE: '#10b981',
    LOC: '#06b6d4',
    PRODUCT: '#f59e0b',
    EVENT: '#ec4899',
    WORK_OF_ART: '#6b7280',
    LAW: '#6b7280',
    DATE: '#6b7280',
    MONEY: '#6b7280',
    PERCENT: '#6b7280',
    QUANTITY: '#6b7280',
    UNKNOWN: '#6b7280',
  };
  return colorMap[type] || '#6b7280';
}
