'use client';

import { useState } from 'react';
import {
  Statement,
  Entity,
  ExtractionMethod,
  getEntityBadgeClass,
} from '@/lib/types';
import {
  ArrowRight,
  Quote,
  ThumbsUp,
  Loader2,
  ChevronDown,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';

interface StatementListProps {
  statements: Statement[];
  onLike?: () => void;
  isLiking?: boolean;
  hasLiked?: boolean;
}

/** Map a canonical_id to an external reference URL, when we recognise the source. */
function canonicalIdLink(canonicalId: string): { url: string; label: string } | null {
  // Accept "SOURCE:ID" or "source:id" forms.
  const [rawSource, ...rest] = canonicalId.split(':');
  const id = rest.join(':');
  if (!rawSource || !id) return null;
  const source = rawSource.toUpperCase();

  switch (source) {
    case 'WIKIDATA':
      return { url: `https://www.wikidata.org/wiki/${id}`, label: 'Wikidata' };
    case 'LEI':
      return { url: `https://search.gleif.org/#/record/${id}`, label: 'GLEIF' };
    case 'SEC':
    case 'SEC-CIK':
    case 'CIK':
      return {
        url: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${id}`,
        label: 'SEC EDGAR',
      };
    case 'CH':
    case 'COMPANIES_HOUSE':
      return {
        url: `https://find-and-update.company-information.service.gov.uk/company/${id}`,
        label: 'Companies House',
      };
    default:
      return null;
  }
}

function EntityBadge({ name, type }: { name: string; type: string }) {
  const badgeClass = getEntityBadgeClass(type as Entity['type']);
  return (
    <span className={`badge ${badgeClass}`}>
      <span className="font-normal mr-1 opacity-70">{type}</span>
      {name}
    </span>
  );
}

function CanonicalIdChip({ canonicalId }: { canonicalId: string }) {
  const link = canonicalIdLink(canonicalId);
  if (link) {
    return (
      <a
        href={link.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200 hover:bg-sky-100"
        title={`Open ${link.label} record`}
      >
        {canonicalId}
        <ExternalLink className="w-2.5 h-2.5" />
      </a>
    );
  }
  return (
    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200">
      {canonicalId}
    </span>
  );
}

/**
 * Compact one-row context summary for a qualified entity. Renders inline
 * facts (role · org · region · legal name · source) followed by the
 * canonical-ID link and match-method/confidence chip. Returns null when
 * there's nothing to say — the caller is responsible for hiding the
 * surrounding label too.
 */
function EntityContextLine({ entity }: { entity: Entity }) {
  const q = entity.qualifiers;
  const facts: string[] = [];
  if (q?.role) facts.push(q.role);
  if (q?.org) facts.push(q.org);
  if (q?.legal_name && q.legal_name !== entity.name) facts.push(q.legal_name);
  const place = q?.city ?? q?.region ?? q?.country ?? q?.jurisdiction;
  if (place) facts.push(place);
  // Extra unscoped identifiers (LEI, ticker, etc.) — show as `key=value`.
  if (q?.identifiers) {
    for (const [k, v] of Object.entries(q.identifiers)) {
      facts.push(`${k}=${v}`);
    }
  }

  const hasMatchInfo = typeof entity.matchConfidence === 'number' && entity.matchMethod;
  const hasOriginalText = entity.text && entity.text !== entity.name;
  const hasAnything = facts.length > 0 || entity.canonicalId || hasMatchInfo || hasOriginalText;
  if (!hasAnything) return null;

  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-gray-600 leading-snug">
      {facts.length > 0 && (
        <span>
          {facts.map((f, i) => (
            <span key={i}>
              {i > 0 && <span className="mx-1.5 text-gray-300">·</span>}
              {f}
            </span>
          ))}
        </span>
      )}
      {entity.canonicalId && <CanonicalIdChip canonicalId={entity.canonicalId} />}
      {hasMatchInfo && (
        <span
          className="text-[10px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 border border-violet-200"
          title="How the entity was matched against the canonical database"
        >
          {entity.matchMethod} · {Math.round((entity.matchConfidence ?? 0) * 100)}%
        </span>
      )}
      {hasOriginalText && (
        <span className="text-gray-400 italic" title="Original span in the source text">
          &ldquo;{entity.text}&rdquo;
        </span>
      )}
    </div>
  );
}

function ConfidenceBadge({ confidence }: { confidence?: number }) {
  if (confidence === undefined || confidence === null) {
    return (
      <span
        className="text-xs font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-500"
        title="Confidence not available"
      >
        —%
      </span>
    );
  }
  const percent = Math.round(confidence * 100);
  let colorClass = 'bg-gray-100 text-gray-600';
  if (confidence >= 0.8) colorClass = 'bg-green-100 text-green-700';
  else if (confidence >= 0.6) colorClass = 'bg-yellow-100 text-yellow-700';
  else colorClass = 'bg-red-100 text-red-700';
  return (
    <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${colorClass}`} title={`Confidence: ${percent}%`}>
      {percent}%
    </span>
  );
}

function ExtractionMethodBadge({ method }: { method?: ExtractionMethod }) {
  const methodLabels: Record<ExtractionMethod, string> = {
    hybrid: 'Hybrid',
    spacy: 'spaCy',
    split: 'Split',
    model: 'Model',
    gliner_relation: 'GLiNER2',
  };
  const methodColors: Record<ExtractionMethod, string> = {
    hybrid: 'bg-blue-100 text-blue-700',
    spacy: 'bg-purple-100 text-purple-700',
    split: 'bg-orange-100 text-orange-700',
    model: 'bg-gray-100 text-gray-600',
    gliner_relation: 'bg-emerald-100 text-emerald-700',
  };
  const methodDescriptions: Record<ExtractionMethod, string> = {
    hybrid: 'Model subject/object + spaCy predicate',
    spacy: 'All components from spaCy parsing',
    split: 'Source text split around predicate',
    model: 'All components from T5-Gemma model',
    gliner_relation: 'GLiNER2 relation extraction',
  };
  if (!method) {
    return (
      <span
        className="text-xs font-medium px-1.5 py-0.5 rounded bg-gray-100 text-gray-500"
        title="Extraction method not specified"
      >
        —
      </span>
    );
  }
  return (
    <span
      className={`text-xs font-medium px-1.5 py-0.5 rounded ${methodColors[method]}`}
      title={methodDescriptions[method]}
    >
      {methodLabels[method]}
    </span>
  );
}

function LabelBadge({
  label,
  value,
  confidence,
}: {
  label: string;
  value: string | number | boolean;
  confidence?: number;
}) {
  let colorClass = 'bg-gray-100 text-gray-600';
  if (label === 'sentiment') {
    if (value === 'positive') colorClass = 'bg-green-100 text-green-700';
    else if (value === 'negative') colorClass = 'bg-red-100 text-red-700';
    else colorClass = 'bg-gray-100 text-gray-600';
  } else if (label === 'relation_type') {
    colorClass = 'bg-indigo-100 text-indigo-700';
  } else if (label === 'confidence') {
    return null; // surfaced separately in the header
  }
  const formattedValue = typeof value === 'number' ? value.toFixed(3) : String(value);
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colorClass}`}>
      <span className="opacity-60">{label}:</span> {formattedValue}
      {typeof confidence === 'number' && confidence < 1 && confidence > 0 && (
        <span className="opacity-50 ml-1">({Math.round(confidence * 100)}%)</span>
      )}
    </span>
  );
}

function TaxonomyBadge({
  category,
  label,
  confidence,
}: {
  category: string;
  label: string;
  confidence: number;
}) {
  const percent = Math.round(confidence * 100);
  return (
    <span
      className="text-xs px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200"
      title={`${category} · ${label} · ${percent}%`}
    >
      <span className="opacity-60 mr-1">{category}:</span>
      {label}
      <span className="opacity-50 ml-1">{percent}%</span>
    </span>
  );
}

/** Helper: does an entity carry any context worth showing under the triple? */
function hasEntityContext(e: Entity): boolean {
  return Boolean(
    e.canonicalId
      || e.qualifiers
      || e.matchMethod
      || (e.text && e.text !== e.name)
      || (e.fqn && e.fqn !== e.name),
  );
}

function StatementCard({ statement, index }: { statement: Statement; index: number }) {
  const [showAll, setShowAll] = useState(false);

  const labelsToShow = (statement.labels || []).filter(
    l => l.label_type !== 'confidence',
  );
  const taxonomy = statement.taxonomyResults || [];
  const visibleTaxonomy = showAll ? taxonomy : taxonomy.slice(0, 3);
  const hiddenTaxonomyCount = Math.max(0, taxonomy.length - visibleTaxonomy.length);
  const canExpand = taxonomy.length > 3;

  const subjectHasContext = hasEntityContext(statement.subject);
  const objectHasContext = Boolean(statement.object.name) && hasEntityContext(statement.object);
  const showContextSection = subjectHasContext || objectHasContext;

  return (
    <div className="editorial-card p-4 space-y-3">
      {/* Triple line — the focal point. Index, type-tinted SVO badges,
          predicate-category chip, and a quiet meta cluster on the right. */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <span className="text-xs font-bold text-gray-400 mt-1">#{index + 1}</span>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 flex-1 min-w-0">
            <EntityBadge name={statement.subject.name} type={statement.subject.type} />
            <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
            <span className="font-semibold text-gray-700">
              {statement.canonicalPredicate || statement.predicate}
              {statement.canonicalPredicate
                && statement.canonicalPredicate !== statement.predicate && (
                  <span
                    className="text-gray-400 font-normal ml-1"
                    title={`Original: "${statement.predicate}"`}
                  >
                    *
                  </span>
                )}
            </span>
            {statement.predicateCategory && (
              <span
                className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200"
                title="Predicate category"
              >
                {statement.predicateCategory}
              </span>
            )}
            {statement.object.name && (
              <>
                <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <EntityBadge name={statement.object.name} type={statement.object.type} />
              </>
            )}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 text-xs flex-shrink-0">
          <ConfidenceBadge confidence={statement.confidence} />
          {statement.extractionMethod && <ExtractionMethodBadge method={statement.extractionMethod} />}
        </div>
      </div>

      {/* Entity context — one tight row per entity that has anything to add.
          Skips silently when neither side is qualified. */}
      {showContextSection && (
        <div className="pl-6 space-y-1.5 border-l-2 border-gray-100 ml-1.5">
          {subjectHasContext && (
            <div>
              <span className="text-[10px] uppercase tracking-wide text-gray-400 mr-2">Subject</span>
              <EntityContextLine entity={statement.subject} />
            </div>
          )}
          {objectHasContext && (
            <div>
              <span className="text-[10px] uppercase tracking-wide text-gray-400 mr-2">Object</span>
              <EntityContextLine entity={statement.object} />
            </div>
          )}
        </div>
      )}

      {/* Labels + topics merged onto one wrapping row when both are small.
          Keep them separate visually only when topics need an expander. */}
      {(labelsToShow.length > 0 || taxonomy.length > 0) && (
        <div className="pl-6 space-y-1.5">
          {labelsToShow.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {labelsToShow.map((label, i) => (
                <LabelBadge
                  key={i}
                  label={label.label_type}
                  value={label.label_value}
                  confidence={label.confidence}
                />
              ))}
            </div>
          )}
          {taxonomy.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {visibleTaxonomy.map((t, i) => (
                <TaxonomyBadge key={i} category={t.category} label={t.label} confidence={t.confidence} />
              ))}
              {canExpand && (
                <button
                  onClick={() => setShowAll(s => !s)}
                  className="inline-flex items-center gap-0.5 text-xs text-gray-500 hover:text-gray-700"
                >
                  {showAll ? (
                    <>
                      <ChevronDown className="w-3 h-3" />
                      show less
                    </>
                  ) : (
                    <>
                      <ChevronRight className="w-3 h-3" />
                      +{hiddenTaxonomyCount} more
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Source text */}
      {statement.text && (
        <div className="flex items-start gap-2 pl-6">
          <Quote className="w-4 h-4 text-gray-300 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-gray-600 italic leading-relaxed">{statement.text}</p>
        </div>
      )}
    </div>
  );
}

export function StatementList({ statements, onLike, isLiking, hasLiked }: StatementListProps) {
  if (statements.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>No statements extracted yet.</p>
        <p className="text-sm mt-2">Enter some text and click &quot;Extract Statements&quot; to begin.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold text-lg">Extracted Statements</h3>
        <span className="text-sm text-gray-500">
          {statements.length} statement{statements.length !== 1 ? 's' : ''}
        </span>
      </div>

      {statements.map((statement, index) => (
        <StatementCard key={index} statement={statement} index={index} />
      ))}

      {onLike && (
        <div className="pt-4 border-t mt-4">
          <button
            onClick={onLike}
            disabled={isLiking || hasLiked}
            className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold transition-all ${
              hasLiked
                ? 'text-green-600 bg-green-50 border border-green-200 cursor-default'
                : isLiking
                ? 'text-gray-400 bg-gray-50 border border-gray-200 cursor-wait'
                : 'text-gray-600 hover:text-green-600 hover:bg-green-50 border border-gray-200 hover:border-green-200'
            }`}
            title={hasLiked ? 'Thanks for the feedback!' : 'Mark extraction as correct'}
          >
            {isLiking ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <ThumbsUp className={`w-4 h-4 ${hasLiked ? 'fill-current' : ''}`} />
            )}
            {hasLiked ? 'Saved!' : 'Looks good'}
          </button>
        </div>
      )}
    </div>
  );
}
