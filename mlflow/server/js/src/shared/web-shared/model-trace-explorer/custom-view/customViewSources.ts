import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { asString, safeJsonStringify } from './catalogPrimitiveUtils';
import { type CustomViewData, getSpanAttributes } from './customViewBuilders';

// A data-binding marker the LLM emits in place of literal data, e.g.
// `{ "$source": "metrics.latency" }` or `{ "$source": "assessments" }`.
export type SourceMarker = { $source: string } & Record<string, unknown>;

// A span-targeting marker the LLM emits in place of a literal `spanId`, so a
// feedback control re-targets the equivalent span in whatever trace is open.
//  - "root"                         -> the trace's root span
//  - { type: "TOOL", nth?: 0 }      -> the nth span of that type (default 0)
//  - { name: "run_sql_query" }      -> the first span whose title matches
export type SpanRefSelector = 'root' | { type?: string; name?: string; nth?: number };
export type SpanRefMarker = { $spanRef: SpanRefSelector };

// Scalar sources resolve to a single display string (StatCard value, etc.).
export const SCALAR_SOURCE_NAMES = [
  'metrics.status',
  'metrics.latency',
  'metrics.totalTokens',
  'metrics.assessments',
] as const;

// Structural sources materialize into a set of child components (one per item)
// whose ids are placed into the host component's `children` array.
export const STRUCTURAL_SOURCE_NAMES = ['assessments'] as const;

// A per-span field source resolves a SINGLE span (selected via a spanRef) to one
// of its fields, re-resolved per trace. It lets a KeyValueViewer / Text bind to a
// specific span's output/input/attributes/name/id without baking a literal — the
// missing capability that previously forced the model to hardcode span data.
//   { "$source": "spanField", "spanRef": { "type": "TOOL", "nth": 0 }, "field": "outputs" }
// The structural fields (inputs/outputs/attributes) may additionally carry a
// "path" — an array of keys/indices drilling into the field's nested JSON, e.g.
//   { ..., "field": "outputs", "path": ["choices", 0, "message", "content"] }
// so a single nested scalar renders as readable text instead of a JSON tree.
export const SPAN_FIELD_SOURCE_NAME = 'spanField';
export const SPAN_FIELD_NAMES = ['inputs', 'outputs', 'attributes', 'name', 'spanId'] as const;
// Fields whose value is a JSON object and therefore supports a nested "path".
export const PATHABLE_SPAN_FIELD_NAMES = ['inputs', 'outputs', 'attributes'] as const;

export type ScalarSourceName = (typeof SCALAR_SOURCE_NAMES)[number];
export type StructuralSourceName = (typeof STRUCTURAL_SOURCE_NAMES)[number];
export type SpanFieldName = (typeof SPAN_FIELD_NAMES)[number];
export type SpanFieldPath = (string | number)[];
export type SpanFieldMarker = {
  $source: 'spanField';
  spanRef: SpanRefSelector;
  field: SpanFieldName;
  path?: SpanFieldPath;
};

const SCALAR_SET = new Set<string>(SCALAR_SOURCE_NAMES);
const STRUCTURAL_SET = new Set<string>(STRUCTURAL_SOURCE_NAMES);
const SPAN_FIELD_SET = new Set<string>(SPAN_FIELD_NAMES);
const PATHABLE_SPAN_FIELD_SET = new Set<string>(PATHABLE_SPAN_FIELD_NAMES);

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

export const isSourceMarker = (value: unknown): value is SourceMarker =>
  isRecord(value) && typeof value['$source'] === 'string';

export const isSpanRefMarker = (value: unknown): value is SpanRefMarker => isRecord(value) && '$spanRef' in value;

export const isScalarSource = (name: string): name is ScalarSourceName => SCALAR_SET.has(name);
export const isStructuralSource = (name: string): name is StructuralSourceName => STRUCTURAL_SET.has(name);
export const isSpanFieldSource = (name: string): boolean => name === SPAN_FIELD_SOURCE_NAME;
export const isKnownSource = (name: string): boolean =>
  isScalarSource(name) || isStructuralSource(name) || isSpanFieldSource(name);

// A `spanField`'s "spanRef" is meant to be a BARE selector ("root" / {type,nth} /
// {name}), but the model frequently wraps it as { "$spanRef": <selector> } (the
// feedback-spanId marker syntax). Accept both by unwrapping the marker form.
export const unwrapSpanRefSelector = (value: unknown): unknown => (isSpanRefMarker(value) ? value.$spanRef : value);

// Validates a `$spanRef` selector against the supported grammar (used by the
// template validator). Raw positional indices are intentionally unsupported as
// they are fragile across traces.
export const isValidSpanRefSelector = (selector: unknown): selector is SpanRefSelector => {
  if (selector === 'root') {
    return true;
  }
  if (!isRecord(selector)) {
    return false;
  }
  const hasType = typeof selector['type'] === 'string' && selector['type'].length > 0;
  const hasName = typeof selector['name'] === 'string' && selector['name'].length > 0;
  if (!hasType && !hasName) {
    return false;
  }
  if ('nth' in selector) {
    const nth = selector['nth'];
    if (typeof nth !== 'number' || !Number.isInteger(nth) || nth < 0) {
      return false;
    }
  }
  return true;
};

// Validates an optional `spanField` "path": a non-empty array of string/number
// segments, only meaningful on the structural fields whose value is an object.
export const isValidSpanFieldPath = (path: unknown, field: string): boolean =>
  PATHABLE_SPAN_FIELD_SET.has(field) &&
  Array.isArray(path) &&
  path.length > 0 &&
  path.every((segment) => typeof segment === 'string' || typeof segment === 'number');

// Validates a `spanField` marker: a valid spanRef selector + a known field name,
// plus (if present) a valid nested "path" on a structural field.
export const isValidSpanFieldMarker = (marker: Record<string, unknown>): boolean => {
  if (!isValidSpanRefSelector(unwrapSpanRefSelector(marker['spanRef']))) {
    return false;
  }
  if (typeof marker['field'] !== 'string' || !SPAN_FIELD_SET.has(marker['field'])) {
    return false;
  }
  if ('path' in marker && marker['path'] !== undefined) {
    return isValidSpanFieldPath(marker['path'], marker['field']);
  }
  return true;
};

// Resolves a scalar source name to the current trace's display string.
export const resolveScalarSource = (name: string, data: CustomViewData): string | undefined => {
  if (name.startsWith('metrics.')) {
    const key = name.slice('metrics.'.length) as keyof CustomViewData['metrics'];
    const value = data.metrics?.[key];
    return value === undefined || value === null ? '' : asString(value);
  }
  return undefined;
};

export const resolveSpanRef = (
  selector: SpanRefSelector,
  nodeMap: Record<string, ModelTraceSpanNode>,
): string | undefined => {
  const nodes = Object.values(nodeMap);
  if (nodes.length === 0) {
    return undefined;
  }
  // Deterministic order so "nth" is stable run-to-run.
  const ordered = [...nodes].sort((a, b) => a.start - b.start);

  if (selector === 'root') {
    const root = ordered.find((node) => !node.parentId);
    return root ? String(root.key) : undefined;
  }

  const { type, name, nth = 0 } = selector;
  const matches = ordered.filter((node) => {
    const typeOk = type ? node.type === type : true;
    const title = typeof node.title === 'string' ? node.title : asString(node.title);
    const nameOk = name ? title === name : true;
    return typeOk && nameOk;
  });
  const picked = matches[nth];
  return picked ? String(picked.key) : undefined;
};

// Walks an array of key/index segments into a value, returning undefined as soon
// as a segment hits a non-indexable step OR a key the container does not OWN.
// The own-property check is deliberate: raw bracket access would let segments
// like "__proto__"/"constructor"/"prototype" traverse the prototype chain and
// surface inherited objects/functions, so invalid selectors must fail closed to
// undefined (which the caller renders as '') instead of leaking runtime internals.
const getByPath = (value: unknown, segments: SpanFieldPath): unknown => {
  let current: unknown = value;
  for (const segment of segments) {
    if (current === null || typeof current !== 'object') {
      return undefined;
    }
    if (Array.isArray(current)) {
      // Only real, in-bounds numeric indices into arrays (excludes "length", etc.).
      const index = typeof segment === 'number' ? segment : Number(segment);
      if (!Number.isInteger(index) || index < 0 || index >= current.length) {
        return undefined;
      }
      current = current[index];
    } else {
      const key = String(segment);
      if (!Object.prototype.hasOwnProperty.call(current, key)) {
        return undefined;
      }
      current = (current as Record<string, unknown>)[key];
    }
  }
  return current;
};

// Serializes a structural span field (inputs/outputs/attributes) for display.
// With no path, preserves the legacy whole-object JSON string. With a path, drills
// into the object: a scalar leaf is returned as raw text (so the renderer shows it
// as prose), a nested object/array as JSON, and a missing path as '' so the view
// degrades gracefully across traces whose shape differs.
const resolveStructuralField = (base: unknown, path?: SpanFieldPath): string => {
  if (!path || path.length === 0) {
    return safeJsonStringify(base);
  }
  const resolved = getByPath(base, path);
  if (resolved === undefined) {
    return '';
  }
  return typeof resolved === 'object' && resolved !== null ? safeJsonStringify(resolved) : asString(resolved);
};

// Resolves a `spanField` marker to a display string for the CURRENT trace: finds
// the span via its spanRef selector, then serializes the requested field.
export const resolveSpanFieldSource = (
  marker: Record<string, unknown>,
  nodeMap: Record<string, ModelTraceSpanNode>,
): string => {
  const field = typeof marker['field'] === 'string' ? marker['field'] : '';
  const selector = unwrapSpanRefSelector(marker['spanRef']);
  const spanId = isValidSpanRefSelector(selector) ? resolveSpanRef(selector, nodeMap) : undefined;
  const span = spanId ? nodeMap[spanId] : undefined;
  // Validate the path's contents (not just Array.isArray) before use, since the
  // marker is unvalidated template data; malformed paths fall back to no path.
  const path = isValidSpanFieldPath(marker['path'], field) ? (marker['path'] as SpanFieldPath) : undefined;
  switch (field) {
    case 'inputs':
      return resolveStructuralField(span?.inputs, path);
    case 'outputs':
      return resolveStructuralField(span?.outputs, path);
    case 'attributes':
      return resolveStructuralField(getSpanAttributes(span), path);
    case 'name':
      return span ? (typeof span.title === 'string' ? span.title : asString(span.title)) : '';
    case 'spanId':
      return spanId ?? '';
    default:
      return '';
  }
};
