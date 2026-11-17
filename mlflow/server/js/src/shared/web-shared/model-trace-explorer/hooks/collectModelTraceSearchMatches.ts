import type { ModelTraceEvent, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { createListFromObject, getEventAttributeKey, getLinkFieldKey } from '../ModelTraceExplorer.utils';

type FieldMatches = { key: string; isKeyMatch: boolean; count: number };
type Section = SearchMatch['section'];
const payloadMatches = new WeakMap<WeakKey, { filter: string; section: Section; fields: FieldMatches[] }>();

const countOccurrences = (text: string, filter: string): number => {
  let count = 0;
  for (let index = text.indexOf(filter); index >= 0; index = text.indexOf(filter, index + filter.length)) count++;
  return count;
};

const countFields = (span: ModelTraceSpanNode, section: Section, filter: string): FieldMatches[] => {
  const payload = span[section];
  const cached = payload && typeof payload === 'object' ? payloadMatches.get(payload) : undefined;
  if (cached?.filter === filter && cached.section === section) return cached.fields;

  const fields: FieldMatches[] = [];
  if (section === 'events' || section === 'links') {
    span[section]?.forEach((entry, index) => {
      for (const [key, value] of Object.entries(entry.attributes ?? {})) {
        fields.push({
          key:
            section === 'events'
              ? getEventAttributeKey((entry as ModelTraceEvent).name, index, key)
              : getLinkFieldKey(index, key),
          isKeyMatch: key.toLowerCase().includes(filter),
          count: countOccurrences(
            JSON.stringify(value, null, section === 'links' ? 2 : undefined).toLowerCase(),
            filter,
          ),
        });
      }
    });
  } else {
    for (const { key, value } of createListFromObject(payload)) {
      fields.push({
        key,
        isKeyMatch: key.toLowerCase().includes(filter),
        count: countOccurrences(value.toLowerCase(), filter),
      });
    }
  }
  // Keep only the last query's counts; rebuilt wrappers must not retain old span objects.
  if (payload && typeof payload === 'object') payloadMatches.set(payload, { filter, section, fields });
  return fields;
};

export const getModelTraceSpanSearchMatches = (span: ModelTraceSpanNode, filter: string): SearchMatch[] => {
  if (!filter) return [];
  const matches: SearchMatch[] = [];
  for (const section of ['inputs', 'outputs', 'attributes', 'events', 'links'] as const) {
    for (const { key, isKeyMatch, count } of countFields(span, section, filter)) {
      if (isKeyMatch) matches.push({ span, section, key, isKeyMatch: true, matchIndex: 0 });
      for (let matchIndex = 0; matchIndex < count; matchIndex++) {
        matches.push({ span, section, key, isKeyMatch: false, matchIndex });
      }
    }
  }
  return matches;
};

/** Collects payload matches between browser tasks, returning a cleanup function for obsolete searches. */
export function collectModelTraceSearchMatches(
  roots: ModelTraceSpanNode[],
  filter: string,
  onComplete: (matches: Map<ModelTraceSpanNode, SearchMatch[]>) => void,
): () => void {
  const pending = [...roots];
  const matches = new Map<ModelTraceSpanNode, SearchMatch[]>();
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const collectBatch = () => {
    if (cancelled) return;
    const deadline = performance.now() + 8;
    while (pending.length > 0) {
      const span = pending.pop();
      if (span && !matches.has(span)) {
        matches.set(span, getModelTraceSpanSearchMatches(span, filter));
        for (const child of span.children ?? []) pending.push(child);
      }
      if (performance.now() >= deadline) break;
    }
    if (pending.length > 0) {
      timer = setTimeout(collectBatch, 0);
    } else {
      onComplete(matches);
    }
  };
  collectBatch();
  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}
