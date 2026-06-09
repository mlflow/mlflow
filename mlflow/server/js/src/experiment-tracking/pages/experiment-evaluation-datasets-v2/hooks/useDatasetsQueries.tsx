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
const v2DatasetsPageQueryKey = (experimentId: string) => ['v2ListDatasetsPage', experimentId] as const;

const RECORDS_PAGE_SIZE = 500;

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

interface OssDatasetRecordSource {
  source_type?: string;
  source_data?: { [key: string]: any } | string;
}

interface OssDatasetRecord {
  dataset_record_id: string;
  dataset_id?: string;
  // The OSS GET handler returns these as parsed dicts (see `DatasetRecord.to_dict()` —
  // it `json.loads`es each field before re-serializing the outer envelope). The upsert
  // endpoint accepts the same dict shape on the way back in, so the adapter passes them
  // through untouched in both directions.
  inputs?: { [key: string]: any };
  expectations?: { [key: string]: any };
  tags?: { [key: string]: string };
  source?: OssDatasetRecordSource;
  source_id?: string;
  source_type?: string;
  created_time?: number;
  last_update_time?: number;
  created_by?: string;
  last_updated_by?: string;
  outputs?: { [key: string]: any };
}

interface GetDatasetRecordsResponse {
  // JSON-stringified array of OssDatasetRecord.
  records: string;
  next_page_token?: string;
}

interface UpsertDatasetRecordsResponse {
  inserted_count: number;
  updated_count: number;
  // Server-assigned ids for the upserted records, in request order. Used by the create flow to
  // bind the side panel to the freshly-created record without a follow-up list refetch.
  record_ids?: string[];
}

interface UpdateDatasetRecordsResponse {
  updated_count: number;
}

const msToIso = (ms?: number): string => (typeof ms === 'number' ? new Date(ms).toISOString() : '');

const orEmpty = (value: { [key: string]: any } | undefined): { [key: string]: any } => value ?? {};

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
 * Decodes the OSS record `source` field into the shape v2 components expect. OSS returns
 * `source` as `{ source_type, source_data: dict }`; v2 wants a discriminated
 * `{ human?, document?, trace? }` union. Unknown source_types yield `undefined` so the UI
 * falls through to its neutral state.
 */
const parseRecordSource = (raw: OssDatasetRecord): DatasetRecord['source'] => {
  const src = raw.source;
  if (!src || !src.source_type) return undefined;
  const data = typeof src.source_data === 'string' ? (parseJSONSafe(src.source_data) ?? {}) : (src.source_data ?? {});
  switch (src.source_type) {
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
  inputs: orEmpty(raw.inputs),
  expectations: orEmpty(raw.expectations),
  tags: orEmpty(raw.tags) as { [key: string]: string },
  source: parseRecordSource(raw),
});

/**
 * Inverse of `ossRecordToUniverse` for writes. The OSS upsert endpoint expects records with
 * dict-shaped `inputs` / `expectations` / `tags` / `source` (the same shape the GET handler
 * returns), so we pass them through unchanged. Fields not present in `record` are omitted
 * so the server can interpret an omitted key as "leave alone" when the upsert handler
 * supports it; callers that need replace-semantics must populate every key explicitly.
 */
const universeRecordToOss = (record: Partial<DatasetRecord>): Partial<OssDatasetRecord> => {
  const out: Partial<OssDatasetRecord> = {};
  if (record.dataset_record_id !== undefined) out.dataset_record_id = record.dataset_record_id;
  if (record.inputs !== undefined) out.inputs = record.inputs;
  if (record.expectations !== undefined) out.expectations = record.expectations;
  if (record.tags !== undefined) out.tags = record.tags;
  if (record.source !== undefined) {
    if (record.source.human) {
      out.source = { source_type: 'HUMAN', source_data: record.source.human };
    } else if (record.source.document) {
      out.source = { source_type: 'DOCUMENT', source_data: record.source.document };
    } else if (record.source.trace) {
      out.source = { source_type: 'TRACE', source_data: record.source.trace };
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
 * input hash — but the upsert response now echoes back the server-assigned `record_ids`, so
 * the create flow can resolve the new id directly instead of refetching and guessing.
 *
 * Resolves the new `dataset_record_id` (from `record_ids[0]`). The single-step "add record"
 * flow uses it to bind the side panel to the just-created record and then autosave edits by id.
 */
export function useCreateDatasetRecordMutation(datasetId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (record: Partial<DatasetRecord>): Promise<DatasetRecord> => {
      const body = {
        dataset_id: datasetId,
        records: JSON.stringify([universeRecordToOss(record)]),
      };
      const response = (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), {
        method: 'POST',
        body,
      })) as UpsertDatasetRecordsResponse;
      const newId = response.record_ids?.[0] ?? record.dataset_record_id ?? '';
      return {
        dataset_record_id: newId,
        create_time: new Date().toISOString(),
        last_update_time: new Date().toISOString(),
        inputs: record.inputs ?? {},
        expectations: record.expectations,
        tags: record.tags,
        source: record.source,
      };
    },
    onSuccess: (created) => {
      // Insert the new record into the list cache immediately so a caller that selects it by id
      // (the single-step add flow) resolves it without waiting for the refetch — otherwise the
      // controller's "selected record not found" guard would briefly close the panel.
      const key = listDatasetRecordsQueryKey(datasetId);
      queryClient.setQueryData<DatasetRecord[]>(key, (old) => {
        const list = old ?? [];
        return list.some((r) => r.dataset_record_id === created.dataset_record_id) ? list : [...list, created];
      });
      queryClient.invalidateQueries({ queryKey: key });
    },
  });
}

/**
 * Update existing records in place by id via the dedicated `PATCH /records` endpoint. Unlike
 * the hash-keyed upsert (which can't change a record's `inputs` without orphaning the old row),
 * this addresses the record by `dataset_record_id`, so it is the write path the autosave flow
 * uses for every edit after a record exists — including `inputs` edits.
 *
 * Each entry names a record and supplies a partial patch; the cache is updated optimistically
 * (and rolled back on error) the same way the upsert mutation handles its writes.
 */
export function useUpdateDatasetRecordMutation(datasetId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (
      updates: Array<{ recordId: string; updates: Partial<DatasetRecord> }>,
    ): Promise<UpdateDatasetRecordsResponse> => {
      const records = updates.map(({ recordId, updates: patch }) =>
        universeRecordToOss({ ...patch, dataset_record_id: recordId }),
      );
      return (await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}/records`), {
        method: 'PATCH',
        body: { dataset_id: datasetId, records: JSON.stringify(records) },
      })) as UpdateDatasetRecordsResponse;
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
        queryClient.setQueryData(
          listDatasetRecordsQueryKey(datasetId),
          current.map((r) => {
            const patch = updatesMap.get(r.dataset_record_id);
            return patch ? { ...r, ...patch, last_update_time: new Date().toISOString() } : r;
          }),
        );
      }
      return { snapshots };
    },
    onError: (_err, _variables, context) => {
      if (!context?.snapshots || context.snapshots.size === 0) return;
      const live = queryClient.getQueryData<DatasetRecord[]>(listDatasetRecordsQueryKey(datasetId));
      if (!live) return;
      queryClient.setQueryData(
        listDatasetRecordsQueryKey(datasetId),
        live.map((r) => context.snapshots.get(r.dataset_record_id) ?? r),
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: listDatasetRecordsQueryKey(datasetId) });
    },
  });
}

export function useDeleteDatasetMutation(experimentId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (datasetId: string): Promise<void> => {
      await fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/datasets/${datasetId}`), { method: 'DELETE' });
    },
    onSuccess: (_data, datasetId) => {
      // Drop the per-dataset cache entry so any open V2 detail view (or a Back-navigation
      // landing on the deleted dataset's URL) sees a fresh 404 rather than the stale
      // metadata returned by `useGetDatasetQuery`. The list page rereads via the paginated
      // key below.
      queryClient.removeQueries({ queryKey: getDatasetQueryKey(datasetId) });
      queryClient.invalidateQueries({ queryKey: v2DatasetsPageQueryKey(experimentId) });
    },
  });
}
