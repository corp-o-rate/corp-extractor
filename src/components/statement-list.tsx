'use client';

import { useState } from 'react';
import {
  Statement,
  Entity,
  EntityQualifiers,
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

function QualifierChips({ qualifiers }: { qualifiers: EntityQualifiers }) {
  const chips: Array<{ k: string; v: string }> = [];
  if (qualifiers.role) chips.push({ k: 'role', v: qualifiers.role });
  if (qualifiers.org) chips.push({ k: 'org', v: qualifiers.org });
  if (qualifiers.legal_name) chips.push({ k: 'legal', v: qualifiers.legal_name });
  if (qualifiers.region) chips.push({ k: 'region', v: qualifiers.region });
  if (qualifiers.city) chips.push({ k: 'city', v: qualifiers.city });
  if (qualifiers.source) chips.push({ k: 'src', v: qualifiers.source });
  if (qualifiers.identifiers) {
    for (const [k, v] of Object.entries(qualifiers.identifiers)) {
      chips.push({ k, v });
    }
  }
  if (chips.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-1">
      {chips.map((c, i) => (
        <span
          key={i}
          className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600"
          title={c.k}
        >
          <span className="opacity-60">{c.k}:</span> {c.v}
        </span>
      ))}
    </div>
  );
}

function EntityDetail({ entity }: { entity: Entity }) {
  const hasDetail = entity.fqn || entity.canonicalId || entity.qualifiers || (entity.text && entity.text !== entity.name);
  if (!hasDetail) return null;
  return (
    <div className="mt-1 ml-1 text-xs text-gray-600">
      {entity.fqn && entity.fqn !== entity.name && (
        <div className="font-medium text-gray-700">{entity.fqn}</div>
      )}
      {entity.text && entity.text !== entity.name && (
        <div className="text-gray-400 italic">&ldquo;{entity.text}&rdquo;</div>
      )}
      <div className="flex flex-wrap items-center gap-1 mt-1">
        {entity.canonicalId && <CanonicalIdChip canonicalId={entity.canonicalId} />}
        {typeof entity.matchConfidence === 'number' && entity.matchMethod && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-700 border border-violet-200"
            title={`Match method: ${entity.matchMethod}`}
          >
            {entity.matchMethod} · {Math.round(entity.matchConfidence * 100)}%
          </span>
        )}
      </div>
      {entity.qualifiers && <QualifierChips qualifiers={entity.qualifiers} />}
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

function StatementCard({ statement, index }: { statement: Statement; index: number }) {
  const [showAll, setShowAll] = useState(false);

  const labelsToShow = (statement.labels || []).filter(
    l => l.label_type !== 'confidence',
  );
  const taxonomy = statement.taxonomyResults || [];
  const visibleTaxonomy = showAll ? taxonomy : taxonomy.slice(0, 3);
  const hiddenTaxonomyCount = Math.max(0, taxonomy.length - visibleTaxonomy.length);
  const canExpand = taxonomy.length > 3;

  return (
    <div className="editorial-card p-4">
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <span className="text-xs font-bold text-gray-400 mt-1">#{index + 1}</span>
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
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
              {statement.object.name && (
                <>
                  <ArrowRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <EntityBadge name={statement.object.name} type={statement.object.type} />
                </>
              )}
              {statement.predicateCategory && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                  {statement.predicateCategory}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-gray-400">Source:</span>
            <ExtractionMethodBadge method={statement.extractionMethod} />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-gray-400">Conf:</span>
            <ConfidenceBadge confidence={statement.confidence} />
          </div>
        </div>
      </div>

      {/* Entity details (FQN, canonical id, qualifiers) — side by side */}
      {(statement.subject.canonicalId
        || statement.subject.fqn
        || statement.subject.qualifiers
        || statement.object.canonicalId
        || statement.object.fqn
        || statement.object.qualifiers) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pl-6 mb-2">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-gray-400 mb-0.5">Subject</div>
            <EntityDetail entity={statement.subject} />
          </div>
          {statement.object.name && (
            <div>
              <div className="text-[10px] uppercase tracking-wide text-gray-400 mb-0.5">Object</div>
              <EntityDetail entity={statement.object} />
            </div>
          )}
        </div>
      )}

      {/* Labels row */}
      {labelsToShow.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pl-6 mb-2">
          <span className="text-[10px] uppercase tracking-wide text-gray-400">Labels:</span>
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

      {/* Taxonomy row */}
      {taxonomy.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pl-6 mb-2">
          <span className="text-[10px] uppercase tracking-wide text-gray-400">Topics:</span>
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

      {/* Source text */}
      {statement.text && (
        <div className="flex items-start gap-2 mt-3 pl-6">
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
