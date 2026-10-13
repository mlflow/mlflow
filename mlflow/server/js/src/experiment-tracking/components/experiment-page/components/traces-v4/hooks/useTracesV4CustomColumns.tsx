import { useCallback, useMemo, type Dispatch, type SetStateAction } from 'react';
import { useIntl } from 'react-intl';
import { CodeIcon, TagIcon } from '@databricks/design-system';
import { type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  getTextColumnMaxSize,
  type GenericColumnOption,
  type TraceTableColumn,
} from '@databricks/web-shared/traces-table';
import { isCustomTraceColumnId, METADATA_COLUMN_PREFIX, TAG_COLUMN_PREFIX } from '../utils/customColumns';

export { isCustomTraceColumnId } from '../utils/customColumns';

const INTERNAL_KEY_PREFIX = 'mlflow.';
const ITEM_COMPONENT_ID = 'mlflow.traces-v4.column-selector.custom-item';
const COLUMN_SIZE = { size: 180, minSize: 100 } as const;

const CustomColumnHeader = ({ kind, name }: { kind: CustomColumnKind; name: string }) => (
  <span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
    {kind === 'tag' ? (
      <TagIcon data-testid="trace-tag-column-icon" css={{ fontSize: 16 }} />
    ) : (
      <CodeIcon data-testid="trace-metadata-column-icon" css={{ fontSize: 16 }} />
    )}
    {name}
  </span>
);

type CustomColumnKind = 'tag' | 'metadata';

interface CustomColumnCandidate {
  id: string;
  key: string;
  kind: CustomColumnKind;
}

const columnId = (kind: CustomColumnKind, key: string) =>
  `${kind === 'tag' ? TAG_COLUMN_PREFIX : METADATA_COLUMN_PREFIX}${key}`;

const valueFor = (trace: ModelTraceInfoV3, candidate: CustomColumnCandidate) =>
  candidate.kind === 'tag' ? trace.tags?.[candidate.key] : trace.trace_metadata?.[candidate.key];

const candidateFromId = (id: string): CustomColumnCandidate | undefined => {
  if (id.startsWith(TAG_COLUMN_PREFIX)) {
    return { id, key: id.slice(TAG_COLUMN_PREFIX.length), kind: 'tag' };
  }
  if (id.startsWith(METADATA_COLUMN_PREFIX)) {
    return {
      id,
      key: id.slice(METADATA_COLUMN_PREFIX.length),
      kind: 'metadata',
    };
  }
  return undefined;
};

const discoverCandidates = (traces: ModelTraceInfoV3[]): CustomColumnCandidate[] => {
  const candidates = new Map<string, CustomColumnCandidate>();
  for (const trace of traces) {
    for (const key of Object.keys(trace.tags ?? {})) {
      if (key && !key.startsWith(INTERNAL_KEY_PREFIX)) {
        const id = columnId('tag', key);
        candidates.set(id, { id, key, kind: 'tag' });
      }
    }
    for (const key of Object.keys(trace.trace_metadata ?? {})) {
      if (key && !key.startsWith(INTERNAL_KEY_PREFIX)) {
        const id = columnId('metadata', key);
        candidates.set(id, { id, key, kind: 'metadata' });
      }
    }
  }
  return [...candidates.values()].sort((left, right) => left.key.localeCompare(right.key));
};

export interface TracesV4CustomColumnGroup {
  selectorOptions: GenericColumnOption[];
  visibleIds: string[];
  toggle: (id: string) => void;
  setAllVisible: (visible: boolean) => void;
}

export interface TracesV4CustomColumns {
  columnDefs: TraceTableColumn[];
  tags: TracesV4CustomColumnGroup;
  metadata: TracesV4CustomColumnGroup;
  visibilityById: Record<string, boolean>;
  setVisibility: (visibility: Record<string, boolean> | undefined) => void;
  toggle: (id: string) => void;
  reset: () => void;
}

/** Discovers user-authored tags and metadata on the current page and exposes them as opt-in columns. */
export const useTracesV4CustomColumns = (
  traces: ModelTraceInfoV3[],
  visibility: Record<string, boolean>,
  setVisibility: Dispatch<SetStateAction<Record<string, boolean>>>,
): TracesV4CustomColumns => {
  const intl = useIntl();
  const discovered = useMemo(() => discoverCandidates(traces), [traces]);
  const candidates = useMemo(() => {
    const byId = new Map(discovered.map((candidate) => [candidate.id, candidate]));
    for (const id of Object.keys(visibility)) {
      const candidate = candidateFromId(id);
      if (visibility[id] === true && candidate && !candidate.key.startsWith(INTERNAL_KEY_PREFIX) && !byId.has(id)) {
        byId.set(id, candidate);
      }
    }
    return [...byId.values()].sort((left, right) => left.key.localeCompare(right.key));
  }, [discovered, visibility]);

  const visibleCandidates = useMemo(
    () => candidates.filter(({ id }) => visibility[id] === true),
    [candidates, visibility],
  );
  const columnDefs = useMemo<TraceTableColumn[]>(
    () =>
      visibleCandidates.map((candidate) => {
        const values = traces.map((trace) => valueFor(trace, candidate)?.toString() ?? '');
        return {
          id: candidate.id,
          ...COLUMN_SIZE,
          maxSize: getTextColumnMaxSize([candidate.key, ...values], COLUMN_SIZE.size),
          // Qualify the menu's accessible label with the field kind so a tag and a metadata field
          // that share a key (e.g. `environment`) don't produce identical labels for screen readers.
          labelText:
            candidate.kind === 'tag'
              ? intl.formatMessage(
                  {
                    defaultMessage: 'Tag: {name}',
                    description: 'Accessible label qualifier for a custom trace tag column header',
                  },
                  { name: candidate.key },
                )
              : intl.formatMessage(
                  {
                    defaultMessage: 'Metadata: {name}',
                    description: 'Accessible label qualifier for a custom trace metadata column header',
                  },
                  { name: candidate.key },
                ),
          header: () => <CustomColumnHeader kind={candidate.kind} name={candidate.key} />,
          cell: ({ row }: { row: { original: ModelTraceInfoV3 } }) =>
            valueFor(row.original, candidate)?.toString() ?? '',
          renderSessionCell: (sessionTraces: ModelTraceInfoV3[]) =>
            [...new Set(sessionTraces.map((trace) => valueFor(trace, candidate)?.toString()).filter(Boolean))].join(
              ', ',
            ),
        };
      }),
    [visibleCandidates, traces, intl],
  );

  const toggle = useCallback(
    (id: string) => setVisibility((current) => ({ ...current, [id]: current[id] !== true })),
    [setVisibility],
  );
  const makeGroup = useCallback(
    (kind: CustomColumnKind): TracesV4CustomColumnGroup => {
      const groupCandidates = candidates.filter((candidate) => candidate.kind === kind);
      return {
        selectorOptions: groupCandidates.map(({ id, key }) => ({ id, label: key, componentId: ITEM_COMPONENT_ID })),
        visibleIds: groupCandidates.filter(({ id }) => visibility[id] === true).map(({ id }) => id),
        toggle,
        setAllVisible: (visible) =>
          setVisibility((current) => ({
            ...current,
            ...Object.fromEntries(groupCandidates.map(({ id }) => [id, visible])),
          })),
      };
    },
    [candidates, visibility, toggle, setVisibility],
  );
  const reset = useCallback(() => setVisibility({}), [setVisibility]);
  const restoreVisibility = useCallback(
    (next: Record<string, boolean> | undefined) =>
      setVisibility(
        Object.fromEntries(
          Object.entries(next ?? {}).filter(
            ([id]) => isCustomTraceColumnId(id) && !candidateFromId(id)?.key.startsWith(INTERNAL_KEY_PREFIX),
          ),
        ),
      ),
    [setVisibility],
  );

  return {
    columnDefs,
    tags: makeGroup('tag'),
    metadata: makeGroup('metadata'),
    visibilityById: visibility,
    setVisibility: restoreVisibility,
    toggle,
    reset,
  };
};
