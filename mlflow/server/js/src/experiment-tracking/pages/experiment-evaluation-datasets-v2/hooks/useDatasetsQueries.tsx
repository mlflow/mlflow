/**
 * Universe-shaped data layer over the OSS MLflow `/ajax-api/3.0/mlflow/datasets/*` REST API.
 *
 * The v2 datasets UI was originally written against a Databricks-internal `managed-evals` API
 * that returns ISO timestamps and parsed `inputs` / `expectations` dictionaries. OSS speaks
 * a different REST shape — ms-since-epoch numbers and JSON-string blobs — so this file owns
 * the translation in one place. Every v2 component that needs dataset/record data imports
 * from here; nothing else in v2 talks to fetchAPI directly.
 */
import { useMutation, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';

export const listDatasetRecordsQueryKey = (datasetId: string) => ['listDatasetRecords', datasetId] as const;
const getDatasetQueryKey = (datasetId: string | undefined) => ['getDataset', datasetId] as const;
const listDatasetsQueryKey = (experimentId: string) => ['listDatasets', experimentId] as const;
const v2DatasetsPageQueryKey = (experimentId: string) => ['v2ListDatasetsPage', experimentId] as const;

const RECORDS_PAGE_SIZE = 500;
const DATASETS_PAGE_SIZE = 500;

export interface Dataset {
  dataset_id: string;
  create_time: string;
  created_by?: string;
  last_update_time?: string;
  last_updated_by?: string;
  digest?: string;
  name?: string;
  schema?: string;
  profile?: string;
  source?: string;
  source_type?: string;
}

export interface DatasetRecord {
  dataset_record_id: string;
  create_time: string;
  created_by?: string;
  last_update_time?: string;
  last_updated_by?: string;
  source?: {
    human?: { user_name: string };
    document?: { doc_uri: string; content: string };
    trace?: { trace_id: string };
  };
  inputs: { [key: string]: any };
  expectations?: { [key: string]: any };
  tags?: { [key: string]: string };
}

/**
 * OSS REST shapes — these are what the `/ajax-api/3.0/mlflow/datasets/*` endpoints emit and
 * accept. Kept private; callers in v2 never see them.
 */
interface OssDataset {
  dataset_id: string;
  name?: string;
  tags?: string;
  schema?: string;
  profile?: string;
  digest?: string;
  created_time?: number;
  last_update_time?: number;
  created_by?: string;
  last_updated_by?: string;
  experiment_ids?: string[];
}

interface OssDatasetRecord {
  dataset_record_id: string;
  dataset_id?: string;
  inputs?: string;
  expectations?: string;
  tags?: string;
  source?: string;
  source_id?: string;
  source_type?: string;
  created_time?: number;
  last_update_time?: number;
  created_by?: string;
  last_updated_by?: string;
  outputs?: string;
}

interface SearchDatasetsResponse {
  datasets?: OssDataset[];
  next_page_token?: string;
}

interface GetDatasetRecordsResponse {
  // JSON-stringified array of OssDatasetRecord.
  records: string;
  next_page_token?: string;
}

interface UpsertDatasetRecordsResponse {
  inserted_count: number;
  updated_count: number;
}

const msToIso = (ms?: number): string => (typeof ms === 'number' ? new Date(ms).toISOString() : '');

const parseDict = (value: string | undefined): { [key: string]: any } => {
  if (!value) return {};
  const parsed = parseJSONSafe(value);
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as { [key: string]: any }) : {};
};

const ossDatasetToUniverse = (raw: OssDataset): Dataset => ({
  dataset_id: raw.dataset_id,
  name: raw.name,
  digest: raw.digest,
  schema: raw.schema,
  profile: raw.profile,
  created_by: raw.created_by,
  last_updated_by: raw.last_updated_by,
  create_time: msToIso(raw.created_time),
  last_update_time: msToIso(raw.last_update_time),
  // OSS datasets carry no dataset-level source — left undefined so consumers fall back to
  // their "no source" rendering (typically an em-dash or a `Source` placeholder).
  source: undefined,
  source_type: undefined,
});

/**
 * Decodes the OSS record `source` blob into the shape v2 components expect. OSS stores
 * `source` as a JSON string with `{ source_type, source_data }`; v2 wants a discriminated
 * `{ human?, document?, trace? }` union. The mapping is best-effort — unknown source_types
 * yield `undefined` so the UI falls through to its neutral state.
 */
const parseRecordSource = (raw: OssDatasetRecord): DatasetRecord['source'] => {
  if (!raw.source) return undefined;
  const parsed = parseJSONSafe(raw.source) as {
    source_type?: string;
    source_data?: string | { [key: string]: any };
  } | null;
  if (!parsed || !parsed.source_type) return undefined;
  const data =
    typeof parsed.source_data === 'string' ? (parseJSONSafe(parsed.source_data) ?? {}) : (parsed.source_data ?? {});
  switch (parsed.source_type) {
    case 'HUMAN':
      return { human: { user_name: (data as any).user_name ?? '' } };
    case 'DOCUMENT':
      return { document: { doc_uri: (data as any).doc_uri ?? '', content: (data as any).content ?? '' } };
    case 'TRACE':
      return { trace: { trace_id: (data as any).trace_id ?? '' } };
    default:
      return undefined;
  }
};

const ossRecordToUniverse = (raw: OssDatasetRecord): DatasetRecord => ({
  dataset_record_id: raw.dataset_record_id,
  created_by: raw.created_by,
  last_updated_by: raw.last_updated_by,
  create_time: msToIso(raw.created_time),
  last_update_time: msToIso(raw.last_update_time),
  inputs: parseDict(raw.inputs),
  expectations: parseDict(raw.expectations),
  tags: parseDict(raw.tags) as { [key: string]: string },
  source: parseRecordSource(raw),
});

/**
 * Inverse of `ossRecordToUniverse` for writes. Universe-shaped fields are re-serialized into
 * the JSON-string blobs OSS expects. Fields not present in `record` are omitted so the server
 * can interpret an omitted key as "leave alone" when the upsert handler supports it; callers
 * that need replace-semantics must populate every key explicitly.
 */
const universeRecordToOss = (record: Partial<DatasetRecord>): Partial<OssDatasetRecord> => {
  const out: Partial<OssDatasetRecord> = {};
  if (record.dataset_record_id !== undefined) out.dataset_record_id = record.dataset_record_id;
  if (record.inputs !== undefined) out.inputs = JSON.stringify(record.inputs);
  if (record.expectations !== undefined) out.expectations = JSON.stringify(record.expectations);
  if (record.tags !== undefined) out.tags = JSON.stringify(record.tags);
  if (record.source !== undefined) {
    if (record.source.human) {
      out.source = JSON.stringify({ source_type: 'HUMAN', source_data: JSON.stringify(record.source.human) });
    } else if (record.source.document) {
      out.source = JSON.stringify({ source_type: 'DOCUMENT', source_data: JSON.stringify(record.source.document) });
    } else if (record.source.trace) {
      out.source = JSON.stringify({ source_type: 'TRACE', source_data: JSON.stringify(record.source.trace) });
    }
  }
  return out;
};

export interface UseGetDatasetQueryOptions {
  retry?: boolean | number;
}

export function useGetDatasetQuery(datasetId?: string, options: UseGetDatasetQueryOptions = {}) {
  return useQuery({
    queryKey: getDatasetQueryKey(datasetId),
    queryFn: async (): Promise<Dataset> => {
      const response = (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}`))) as {
        dataset?: OssDataset;
      };
      if (!response.dataset) {
        throw new Error(`Dataset ${datasetId} not found`);
      }
      return ossDatasetToUniverse(response.dataset);
    },
    cacheTime: 5 * 60 * 1000,
    staleTime: 2 * 60 * 1000,
    refetchOnWindowFocus: false,
    retry: options.retry ?? 3,
    enabled: Boolean(datasetId),
  });
}

/**
 * Eagerly fetches every dataset for `experimentId` by walking `next_page_token` in the query
 * function. v2 also has its own paginated `useDatasetsPageQuery` for the new list page — this
 * hook is preserved only for callers that still expect the legacy flat-list semantics
 * (`useCreateDatasetMutation`'s cache write, primarily).
 */
export function useListDatasetsQuery(experimentId: string) {
  return useQuery({
    queryKey: listDatasetsQueryKey(experimentId),
    queryFn: async (): Promise<Dataset[]> => {
      const out: Dataset[] = [];
      let pageToken: string | undefined;
      do {
        const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/datasets/search'), {
          method: 'POST',
          body: {
            experiment_ids: [experimentId],
            max_results: DATASETS_PAGE_SIZE,
            order_by: ['created_time DESC'],
            page_token: pageToken,
          },
        })) as SearchDatasetsResponse;
        response.datasets?.forEach((d) => out.push(ossDatasetToUniverse(d)));
        pageToken = response.next_page_token;
      } while (pageToken);
      return out;
    },
    cacheTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Eagerly fetches every record for `datasetId`. v2 wraps this in `useDatasetRecordsPageQuery`
 * to slice the flat list into client-side pages; that page-slicer is what the records table
 * reads. Cache key `['listDatasetRecords', datasetId]` is shared with mutations below for
 * optimistic write coordination.
 */
export function useListDatasetRecordsQuery(datasetId: string) {
  return useQuery({
    queryKey: listDatasetRecordsQueryKey(datasetId),
    queryFn: async (): Promise<DatasetRecord[]> => {
      const out: DatasetRecord[] = [];
      let pageToken: string | undefined;
      do {
        const params = new URLSearchParams({ dataset_id: datasetId, max_results: String(RECORDS_PAGE_SIZE) });
        if (pageToken) params.set('page_token', pageToken);
        const response = (await fetchAPI(
          getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records?${params.toString()}`),
        )) as GetDatasetRecordsResponse;
        const raw = (parseJSONSafe(response.records) as OssDatasetRecord[]) ?? [];
        raw.forEach((r) => out.push(ossRecordToUniverse(r)));
        pageToken = response.next_page_token;
      } while (pageToken);
      return out;
    },
    cacheTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
    retry: false,
  });
}

/**
 * Universe's mutation surface treats deletes as a batch operation taking record IDs and
 * propagating to the cached record list optimistically. OSS exposes the same DELETE endpoint
 * (`DELETE /mlflow/datasets/{id}/records` with `{ dataset_record_ids: [...] }`), so the
 * adapter is a straight pass-through plus the optimistic dance.
 */
export function useDeleteDatasetRecordsMutation(datasetId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (recordIds: string[]): Promise<void> => {
      await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), {
        method: 'DELETE',
        body: { dataset_record_ids: recordIds },
      });
    },
    onMutate: async (recordIds) => {
      await queryClient.cancelQueries(listDatasetRecordsQueryKey(datasetId));
      const current = queryClient.getQueryData<DatasetRecord[]>(listDatasetRecordsQueryKey(datasetId));
      const idSet = new Set(recordIds);
      const snapshots = new Map<string, DatasetRecord>();
      current?.forEach((r) => {
        if (idSet.has(r.dataset_record_id)) snapshots.set(r.dataset_record_id, r);
      });
      if (current) {
        queryClient.setQueryData(
          listDatasetRecordsQueryKey(datasetId),
          current.filter((r) => !idSet.has(r.dataset_record_id)),
        );
      }
      return { snapshots };
    },
    onError: (_err, _variables, context) => {
      if (!context?.snapshots || context.snapshots.size === 0) return;
      const live = queryClient.getQueryData<DatasetRecord[]>(listDatasetRecordsQueryKey(datasetId)) ?? [];
      const liveIds = new Set(live.map((r) => r.dataset_record_id));
      const reinserted = Array.from(context.snapshots.values()).filter((r) => !liveIds.has(r.dataset_record_id));
      if (reinserted.length === 0) return;
      queryClient.setQueryData(listDatasetRecordsQueryKey(datasetId), [...live, ...reinserted]);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: listDatasetRecordsQueryKey(datasetId) });
    },
  });
}

/**
 * Patch-style upsert. Each entry names a record by id and supplies a partial update; the
 * adapter merges the patch onto the current cached record before sending so the OSS
 * upsert endpoint (which replaces by `dataset_record_id`) sees a complete record. The
 * cache merge also drives the optimistic update.
 */
export function useUpsertDatasetRecordsMutation(datasetId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (
      updates: Array<{
        recordId: string;
        updates: Partial<DatasetRecord>;
        updateMask?: Partial<DatasetRecord>;
      }>,
    ): Promise<UpsertDatasetRecordsResponse> => {
      const current = queryClient.getQueryData<DatasetRecord[]>(listDatasetRecordsQueryKey(datasetId)) ?? [];
      const byId = new Map(current.map((r) => [r.dataset_record_id, r] as const));
      const merged = updates.map(({ recordId, updates }) => {
        const base = byId.get(recordId);
        const next: Partial<DatasetRecord> = { ...(base ?? {}), ...updates, dataset_record_id: recordId };
        return universeRecordToOss(next);
      });
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), {
        method: 'POST',
        body: { dataset_id: datasetId, records: JSON.stringify(merged) },
      })) as UpsertDatasetRecordsResponse;
    },
    onMutate: async (updates) => {
      await queryClient.cancelQueries(listDatasetRecordsQueryKey(datasetId));
      const current = queryClient.getQueryData<DatasetRecord[]>(listDatasetRecordsQueryKey(datasetId));
      const touchedIds = new Set(updates.map((u) => u.recordId));
      const snapshots = new Map<string, DatasetRecord>();
      current?.forEach((r) => {
        if (touchedIds.has(r.dataset_record_id)) snapshots.set(r.dataset_record_id, r);
      });
      if (current) {
        const updatesMap = new Map(updates.map((u) => [u.recordId, u.updates] as const));
        const updatedRecords = current.map((r) => {
          const patch = updatesMap.get(r.dataset_record_id);
          return patch ? { ...r, ...patch, last_update_time: new Date().toISOString() } : r;
        });
        queryClient.setQueryData(listDatasetRecordsQueryKey(datasetId), updatedRecords);
      }
      return { snapshots };
    },
    onError: (_err, _variables, context) => {
      if (!context?.snapshots || context.snapshots.size === 0) return;
      const live = queryClient.getQueryData<DatasetRecord[]>(listDatasetRecordsQueryKey(datasetId));
      if (!live) return;
      const restored = live.map((r) => context.snapshots.get(r.dataset_record_id) ?? r);
      queryClient.setQueryData(listDatasetRecordsQueryKey(datasetId), restored);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: listDatasetRecordsQueryKey(datasetId) });
    },
  });
}

/**
 * Single-record create. OSS has no dedicated create endpoint — `POST /records` upserts by
 * id and emits inserted/updated counts. The adapter sends a one-element array and then
 * refetches the records list so the new server-assigned id is visible to the table.
 *
 * Returns the input record echoed back (without a server id), because OSS upsert response
 * carries only counts. Universe callers that need the new id read it from the refetched
 * list cache rather than from this hook's resolved value.
 */
export function useCreateDatasetRecordMutation(datasetId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (record: Partial<DatasetRecord>): Promise<DatasetRecord> => {
      const body = {
        dataset_id: datasetId,
        records: JSON.stringify([universeRecordToOss(record)]),
      };
      await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), {
        method: 'POST',
        body,
      });
      // Universe consumers ignore the resolved value beyond its existence; a zero-filled
      // shell here is good enough until the followup refetch lands the real record.
      return {
        dataset_record_id: record.dataset_record_id ?? '',
        create_time: new Date().toISOString(),
        last_update_time: new Date().toISOString(),
        inputs: record.inputs ?? {},
        expectations: record.expectations,
        tags: record.tags,
        source: record.source,
      };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listDatasetRecordsQueryKey(datasetId) });
    },
  });
}

export function useCreateDatasetMutation(experimentId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ name, experimentIds }: { name: string; experimentIds?: string[] }): Promise<Dataset> => {
      const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/datasets/create'), {
        method: 'POST',
        body: { name, experiment_ids: experimentIds ?? [experimentId] },
      })) as { dataset: OssDataset };
      return ossDatasetToUniverse(response.dataset);
    },
    onSuccess: (newDataset) => {
      queryClient.setQueryData<Dataset[]>(listDatasetsQueryKey(experimentId), (prev) =>
        prev ? [newDataset, ...prev] : [newDataset],
      );
      queryClient.invalidateQueries({ queryKey: v2DatasetsPageQueryKey(experimentId) });
    },
  });
}

export function useDeleteDatasetMutation(experimentId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (datasetId: string): Promise<void> => {
      await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}`), { method: 'DELETE' });
    },
    onSuccess: async (_data, datasetId) => {
      // Drop the per-dataset cache entry alongside the two list keys so any open V2 detail
      // view (or a Back-navigation landing on the deleted dataset's URL) sees a fresh 404
      // rather than the stale metadata returned by `useGetDatasetQuery`.
      queryClient.removeQueries({ queryKey: getDatasetQueryKey(datasetId) });
      queryClient.setQueryData<Dataset[]>(
        listDatasetsQueryKey(experimentId),
        (prev) => prev?.filter((d) => d.dataset_id !== datasetId) ?? prev,
      );
      queryClient.invalidateQueries({ queryKey: listDatasetsQueryKey(experimentId) });
      queryClient.invalidateQueries({ queryKey: v2DatasetsPageQueryKey(experimentId) });
    },
  });
}
