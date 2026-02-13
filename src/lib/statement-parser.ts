/**
 * Parse statements from T5-Gemma 2 model output
 *
 * Supports two formats:
 *
 * 1. JSON format (v0.2.0+, preferred) - includes confidence scores:
 * {
 *   "statements": [{
 *     "subject": {"text": "Apple Inc.", "type": "ORG"},
 *     "predicate": "committed to",
 *     "object": {"text": "carbon neutral by 2030", "type": "EVENT"},
 *     "source_text": "Apple Inc. committed to becoming carbon neutral by 2030.",
 *     "confidence_score": 0.85,
 *     "canonical_predicate": null
 *   }],
 *   "source_text": "..."
 * }
 *
 * 2. XML format (legacy):
 * <statements>
 *   <stmt>
 *     <subject type="ORG">Apple Inc.</subject>
 *     <object type="EVENT">carbon neutral by 2030</object>
 *     <predicate>committed to</predicate>
 *     <text>Apple Inc. committed to becoming carbon neutral by 2030.</text>
 *   </stmt>
 * </statements>
 */

import { Statement, EntityType, Entity, ExtractionMethod, StatementLabel, TaxonomyResult } from './types';

// Type for the JSON format from the library
interface LibraryEntity {
  text: string;
  type: string;
}

interface LibraryLabel {
  label_type: string;
  label_value: string | number | boolean;
  confidence: number;
  labeler?: string;
}

interface LibraryTaxonomyResult {
  taxonomy_name: string;
  category: string;
  label: string;
  label_id?: number;
  confidence: number;
  classifier?: string;
}

interface LibraryStatement {
  subject: LibraryEntity;
  predicate: string;
  predicate_category?: string | null;
  object: LibraryEntity;
  source_text?: string | null;
  confidence_score?: number | null;
  canonical_predicate?: string | null;
  evidence_span?: [number, number] | null;
  extraction_method?: string | null;
  labels?: LibraryLabel[];
  taxonomy_results?: LibraryTaxonomyResult[];
  // Also handle the as_dict() format from LabeledStatement
  taxonomy?: Array<{ category: string; label: string; confidence: number }>;
}

interface LibraryExtractionResult {
  statements: LibraryStatement[];
  labeled_statements?: LibraryLabeledStatement[];
  source_text?: string | null;
}

// LabeledStatement from pipeline output (as_dict format)
interface LibraryLabeledStatement {
  subject: {
    text: string;
    type: string;
    fqn: string;
    canonical_id?: string | null;
  };
  predicate: string;
  object: {
    text: string;
    type: string;
    fqn: string;
    canonical_id?: string | null;
  };
  source_text?: string | null;
  labels?: Record<string, string | number | boolean>;
  taxonomy?: Array<{ category: string; label: string; confidence: number }>;
}

// LabeledStatement from pipeline output (model_dump format)
interface ModelDumpLabeledStatement {
  statement: {
    subject: { text: string; type: string; entity_ref?: string };
    predicate: string;
    object: { text: string; type: string; entity_ref?: string };
    source_text?: string | null;
    predicate_category?: string | null;
    confidence_score?: number | null;
    canonical_predicate?: string | null;
    extraction_method?: string | null;
  };
  subject_canonical: {
    name: string;
    fqn: string;
    qualifiers?: Array<{ identifier_type: string; identifier_value: string }>;
  };
  object_canonical: {
    name: string;
    fqn: string;
    qualifiers?: Array<{ identifier_type: string; identifier_value: string }>;
  };
  labels: Array<{ label_type: string; label_value: string | number | boolean; confidence: number; labeler?: string }>;
  taxonomy_results: Array<{ taxonomy_name: string; category: string; label: string; label_id?: number; confidence: number; classifier?: string }>;
}

/**
 * Parse entity type from string, with fallback to UNKNOWN
 */
function parseEntityType(typeStr: string | null): EntityType {
  if (!typeStr) return 'UNKNOWN';

  const normalized = typeStr.toUpperCase().trim();
  const validTypes: EntityType[] = [
    'ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
    'WORK_OF_ART', 'LAW', 'DATE', 'MONEY', 'PERCENT', 'QUANTITY'
  ];

  if (validTypes.includes(normalized as EntityType)) {
    return normalized as EntityType;
  }

  return 'UNKNOWN';
}

/**
 * Parse extraction method from string
 */
function parseExtractionMethod(methodStr: string | null | undefined): ExtractionMethod | undefined {
  if (!methodStr) return undefined;

  const normalized = methodStr.toLowerCase().trim();
  const validMethods: ExtractionMethod[] = ['hybrid', 'spacy', 'split', 'model'];

  if (validMethods.includes(normalized as ExtractionMethod)) {
    return normalized as ExtractionMethod;
  }

  return undefined;
}

/**
 * Extract text content and type from an entity element
 */
function parseEntity(element: Element | null): Entity {
  if (!element) {
    return { name: '', type: 'UNKNOWN' };
  }

  const type = parseEntityType(element.getAttribute('type'));
  const name = element.textContent?.trim() || '';

  return { name, type };
}

/**
 * Parse a single statement element
 */
function parseStatementElement(stmtElement: Element): Statement | null {
  const subjectEl = stmtElement.querySelector('subject');
  const objectEl = stmtElement.querySelector('object');
  const predicateEl = stmtElement.querySelector('predicate');
  const textEl = stmtElement.querySelector('text');

  const subject = parseEntity(subjectEl);
  const object = parseEntity(objectEl);
  const predicate = predicateEl?.textContent?.trim() || '';
  const text = textEl?.textContent?.trim() || '';

  // Skip statements with missing required fields
  if (!subject.name || !predicate) {
    return null;
  }

  return {
    subject,
    object,
    predicate,
    text: text || `${subject.name} ${predicate} ${object.name}`.trim(),
  };
}

/**
 * Create a normalized key for deduplication
 * Uses lowercase trimmed subject name + predicate + object name
 */
function getStatementKey(statement: Statement): string {
  const subject = statement.subject.name.toLowerCase().trim();
  const predicate = statement.predicate.toLowerCase().trim();
  const object = statement.object.name.toLowerCase().trim();
  return `${subject}|||${predicate}|||${object}`;
}

/**
 * Remove duplicate statements based on subject-predicate-object triples
 * Keeps the first occurrence of each unique triple
 */
function deduplicateStatements(statements: Statement[]): Statement[] {
  const seen = new Set<string>();
  const unique: Statement[] = [];

  for (const statement of statements) {
    const key = getStatementKey(statement);
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(statement);
    }
  }

  return unique;
}

/**
 * Parse statements from JSON or XML string
 * Automatically detects format and parses accordingly
 */
export function parseStatements(input: string | LibraryExtractionResult): Statement[] {
  // Handle empty or invalid input
  if (!input) {
    return [];
  }

  // If input is already an object (JSON parsed), handle directly
  if (typeof input === 'object') {
    return parseJsonStatements(input);
  }

  const trimmed = input.trim();

  // Try to detect JSON format (starts with { or has "statements" key)
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed);
      return parseJsonStatements(parsed);
    } catch {
      // Not valid JSON, fall through to XML parsing
      console.warn('Failed to parse as JSON, trying XML');
    }
  }

  // Parse as XML (legacy format)
  return parseXmlStatements(trimmed);
}

/**
 * Parse statements from JSON format (v0.2.0+)
 */
function parseJsonStatements(data: LibraryExtractionResult | LibraryStatement[]): Statement[] {
  // Handle labeled_statements from pipeline output (model_dump or as_dict format)
  if (!Array.isArray(data) && data.labeled_statements && data.labeled_statements.length > 0) {
    // Detect model_dump format: items have a `statement` sub-object
    const first = data.labeled_statements[0];
    if (first && 'statement' in first && typeof first.statement === 'object') {
      return parseModelDumpLabeledStatements(data.labeled_statements as unknown as ModelDumpLabeledStatement[]);
    }
    return parseLabeledStatements(data.labeled_statements);
  }

  // Handle array of statements directly
  const statements = Array.isArray(data) ? data : data.statements || [];

  return statements.map((stmt: LibraryStatement) => {
    // Parse labels
    const labels: StatementLabel[] = (stmt.labels || []).map(l => ({
      label_type: l.label_type,
      label_value: l.label_value,
      confidence: l.confidence,
      labeler: l.labeler,
    }));

    // Parse taxonomy results (handle both formats)
    let taxonomyResults: TaxonomyResult[] = [];
    if (stmt.taxonomy_results && stmt.taxonomy_results.length > 0) {
      taxonomyResults = stmt.taxonomy_results.map(t => ({
        taxonomy_name: t.taxonomy_name,
        category: t.category,
        label: t.label,
        label_id: t.label_id,
        confidence: t.confidence,
        classifier: t.classifier,
      }));
    } else if (stmt.taxonomy && stmt.taxonomy.length > 0) {
      // Handle simplified as_dict() format
      taxonomyResults = stmt.taxonomy.map(t => ({
        taxonomy_name: 'esg_topics',
        category: t.category,
        label: t.label,
        confidence: t.confidence,
      }));
    }

    return {
      subject: {
        name: stmt.subject?.text || '',
        type: parseEntityType(stmt.subject?.type || null),
      },
      object: {
        name: stmt.object?.text || '',
        type: parseEntityType(stmt.object?.type || null),
      },
      predicate: stmt.predicate || '',
      predicateCategory: stmt.predicate_category ?? undefined,
      text: stmt.source_text || `${stmt.subject?.text || ''} ${stmt.predicate || ''} ${stmt.object?.text || ''}`.trim(),
      confidence: stmt.confidence_score ?? undefined,
      canonicalPredicate: stmt.canonical_predicate ?? undefined,
      extractionMethod: parseExtractionMethod(stmt.extraction_method),
      labels: labels.length > 0 ? labels : undefined,
      taxonomyResults: taxonomyResults.length > 0 ? taxonomyResults : undefined,
    };
  }).filter((stmt: Statement) => stmt.subject.name && stmt.predicate);
}

/**
 * Parse labeled statements from pipeline output (as_dict format)
 */
function parseLabeledStatements(labeledStmts: LibraryLabeledStatement[]): Statement[] {
  return labeledStmts.map((stmt) => {
    // Convert labels dict to array
    const labels: StatementLabel[] = [];
    if (stmt.labels) {
      for (const [label_type, label_value] of Object.entries(stmt.labels)) {
        labels.push({
          label_type,
          label_value,
          confidence: 1.0, // Labels from as_dict don't include confidence
        });
      }
    }

    // Parse taxonomy
    const taxonomyResults: TaxonomyResult[] = (stmt.taxonomy || []).map(t => ({
      taxonomy_name: 'esg_topics',
      category: t.category,
      label: t.label,
      confidence: t.confidence,
    }));

    return {
      subject: {
        name: stmt.subject?.text || '',
        type: parseEntityType(stmt.subject?.type || null),
      },
      object: {
        name: stmt.object?.text || '',
        type: parseEntityType(stmt.object?.type || null),
      },
      predicate: stmt.predicate || '',
      text: stmt.source_text || `${stmt.subject?.text || ''} ${stmt.predicate || ''} ${stmt.object?.text || ''}`.trim(),
      labels: labels.length > 0 ? labels : undefined,
      taxonomyResults: taxonomyResults.length > 0 ? taxonomyResults : undefined,
    };
  }).filter((stmt: Statement) => stmt.subject.name && stmt.predicate);
}

/**
 * Parse labeled statements from pipeline output (model_dump format)
 *
 * This handles the Pydantic model_dump() format where each labeled statement
 * has nested `statement`, `subject_canonical`, `object_canonical`, `labels`,
 * and `taxonomy_results` fields.
 */
function parseModelDumpLabeledStatements(labeledStmts: ModelDumpLabeledStatement[]): Statement[] {
  return labeledStmts.map((item) => {
    const stmt = item.statement;

    // Use canonical name if available, otherwise fall back to statement text
    const subjectName = item.subject_canonical?.name || stmt.subject?.text || '';
    const objectName = item.object_canonical?.name || stmt.object?.text || '';

    // Parse labels array
    const labels: StatementLabel[] = (item.labels || []).map(l => ({
      label_type: l.label_type,
      label_value: l.label_value,
      confidence: l.confidence,
      labeler: l.labeler,
    }));

    // Parse taxonomy results array
    const taxonomyResults: TaxonomyResult[] = (item.taxonomy_results || []).map(t => ({
      taxonomy_name: t.taxonomy_name,
      category: t.category,
      label: t.label,
      label_id: t.label_id,
      confidence: t.confidence,
      classifier: t.classifier,
    }));

    return {
      subject: {
        name: subjectName,
        type: parseEntityType(stmt.subject?.type || null),
      },
      object: {
        name: objectName,
        type: parseEntityType(stmt.object?.type || null),
      },
      predicate: stmt.predicate || '',
      predicateCategory: stmt.predicate_category ?? undefined,
      text: stmt.source_text || `${subjectName} ${stmt.predicate || ''} ${objectName}`.trim(),
      confidence: stmt.confidence_score ?? undefined,
      canonicalPredicate: stmt.canonical_predicate ?? undefined,
      extractionMethod: parseExtractionMethod(stmt.extraction_method),
      labels: labels.length > 0 ? labels : undefined,
      taxonomyResults: taxonomyResults.length > 0 ? taxonomyResults : undefined,
    };
  }).filter((stmt: Statement) => stmt.subject.name && stmt.predicate);
}

/**
 * Parse statements from XML format (legacy)
 */
function parseXmlStatements(xmlString: string): Statement[] {
  const statements: Statement[] = [];

  // Check for valid XML structure
  if (!xmlString.includes('<statements>') && !xmlString.includes('<stmt>')) {
    console.warn('No <statements> or <stmt> tags found in output');
    return statements;
  }

  try {
    // Wrap in root if needed
    let xmlToParse = xmlString;
    if (!xmlString.startsWith('<statements>')) {
      xmlToParse = `<statements>${xmlString}</statements>`;
    }

    // Parse using DOMParser (works in browser and Node with jsdom)
    const parser = new DOMParser();
    const doc = parser.parseFromString(xmlToParse, 'text/xml');

    // Check for parse errors
    const parseError = doc.querySelector('parsererror');
    if (parseError) {
      console.error('XML parse error:', parseError.textContent);
      // Try fallback regex-based parsing
      return deduplicateStatements(parseStatementsRegex(xmlString));
    }

    // Extract all statement elements
    const stmtElements = doc.querySelectorAll('stmt');

    for (const stmtEl of stmtElements) {
      const statement = parseStatementElement(stmtEl);
      if (statement) {
        statements.push(statement);
      }
    }
  } catch (error) {
    console.error('Error parsing XML statements:', error);
    // Try fallback regex-based parsing
    return deduplicateStatements(parseStatementsRegex(xmlString));
  }

  // Deduplicate before returning
  return deduplicateStatements(statements);
}

/**
 * Fallback regex-based parser for malformed XML
 */
function parseStatementsRegex(xmlString: string): Statement[] {
  const statements: Statement[] = [];

  // Match individual statements
  const stmtRegex = /<stmt>([\s\S]*?)<\/stmt>/g;
  let match;

  while ((match = stmtRegex.exec(xmlString)) !== null) {
    const stmtContent = match[1];

    // Extract fields
    const subjectMatch = stmtContent.match(/<subject(?:\s+type="([^"]*)")?>([\s\S]*?)<\/subject>/);
    const objectMatch = stmtContent.match(/<object(?:\s+type="([^"]*)")?>([\s\S]*?)<\/object>/);
    const predicateMatch = stmtContent.match(/<predicate>([\s\S]*?)<\/predicate>/);
    const textMatch = stmtContent.match(/<text>([\s\S]*?)<\/text>/);

    const subject: Entity = {
      name: subjectMatch?.[2]?.trim() || '',
      type: parseEntityType(subjectMatch?.[1] || null),
    };

    const object: Entity = {
      name: objectMatch?.[2]?.trim() || '',
      type: parseEntityType(objectMatch?.[1] || null),
    };

    const predicate = predicateMatch?.[1]?.trim() || '';
    const text = textMatch?.[1]?.trim() || '';

    if (subject.name && predicate) {
      statements.push({
        subject,
        object,
        predicate,
        text: text || `${subject.name} ${predicate} ${object.name}`.trim(),
      });
    }
  }

  return statements;
}

/**
 * Convert statements to graph data for visualization
 */
export function statementsToGraphData(statements: Statement[]) {
  const nodesMap = new Map<string, { name: string; type: EntityType }>();
  const links: Array<{ source: string; target: string; predicate: string }> = [];

  for (const stmt of statements) {
    // Add subject node
    const subjectId = `${stmt.subject.type}:${stmt.subject.name}`;
    if (!nodesMap.has(subjectId)) {
      nodesMap.set(subjectId, {
        name: stmt.subject.name,
        type: stmt.subject.type,
      });
    }

    // Add object node (if it has a name)
    if (stmt.object.name) {
      const objectId = `${stmt.object.type}:${stmt.object.name}`;
      if (!nodesMap.has(objectId)) {
        nodesMap.set(objectId, {
          name: stmt.object.name,
          type: stmt.object.type,
        });
      }

      // Add link
      links.push({
        source: subjectId,
        target: objectId,
        predicate: stmt.predicate,
      });
    }
  }

  const nodes = Array.from(nodesMap.entries()).map(([id, data]) => ({
    id,
    ...data,
  }));

  return { nodes, links };
}
