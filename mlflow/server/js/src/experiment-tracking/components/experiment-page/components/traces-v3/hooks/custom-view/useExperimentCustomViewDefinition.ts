import { useCallback, useMemo } from 'react';

import { useMutation, useQuery, useQueryClient } from '@databricks/web-shared/query-client';
import {
  type CustomView,
  CUSTOM_VIEW_PREFIX_V1,
  CUSTOM_VIEW_TAG_PREFIX,
  CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES,
  getUtf8ByteLength,
  parseCustomView,
  serializeCustomView,
  viewTagKey,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

import { MlflowService } from '../../../../../../sdk/MlflowService';
import {
  isTextCompressedDeflate,
  textCompressDeflate,
  textDecompressDeflate,
} from '../../../../../../../common/utils/StringUtils';

type ExperimentTag = { key: string; value: string };

const makeUnreadableView = (id: string, name?: string): CustomView => {
  const displayName = typeof name === 'string' && name.trim() ? name : '';
  return { id, name: displayName, label: displayName, instruction: '', template: [], createdAtMs: 0, unreadable: true };
};

// Reads a stored tag value (deflate-aware) into a CustomView. A value that can't
// be read into a valid view is NOT dropped: it becomes an `unreadable`
// placeholder view keyed by the tag id, so an incompatible/corrupt saved view
// stays listed rather than silently vanishing.
const deserializeView = async (id: string, value: string): Promise<CustomView> => {
  let raw: unknown;
  try {
    const json = isTextCompressedDeflate(value) ? await textDecompressDeflate(value) : value;
    raw = JSON.parse(json);
  } catch {
    // Unparseable bytes (bad deflate / malformed JSON): nothing to salvage.
    return makeUnreadableView(id);
  }
  // Salvage a display name from the parsed object up front (cheap, no recursion),
  // so a throw inside parseCustomView can still fall back to a named placeholder.
  const salvagedName =
    raw && typeof raw === 'object' && 'name' in raw && typeof (raw as { name?: unknown }).name === 'string'
      ? (raw as { name: string }).name
      : undefined;
  try {
    const parsed = parseCustomView(raw);
    if (parsed) {
      // The tag-key suffix is the authoritative id (save key off it).
      return { ...parsed, id };
    }
  } catch {
    // parseCustomView is only a shape narrower now (template validation is
    // deferred to render), so it shouldn't throw — but it walks untrusted parsed
    // JSON, so contain any unforeseen throw to a single unreadable placeholder
    // rather than letting it reject Promise.all and drop every saved view.
  }
  // Parsed to a value but not a valid CustomView shape (or threw during parse):
  // keep it as an unreadable placeholder, salvaging a display name when present.
  return makeUnreadableView(id, salvagedName);
};

// Serializes a view for storage, compressing only when raw JSON exceeds the
// safe client limit. Throws when the compressed value also exceeds it.
const serializeViewForTag = async (view: CustomView): Promise<string> => {
  const raw = serializeCustomView(view);
  const rawByteLength = getUtf8ByteLength(raw);
  if (rawByteLength <= CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES) {
    return raw;
  }
  const compressed = await textCompressDeflate(raw);
  const compressedByteLength = getUtf8ByteLength(compressed);
  if (compressedByteLength <= CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES) {
    return compressed;
  }
  throw new Error('This custom view is too large to save. Try simplifying it (fewer or smaller components).');
};

export type ExperimentCustomViewDefinition = {
  views: CustomView[];
  isLoaded: boolean;
  // Undefined (no experiment scope) → the host falls back to a session-local,
  // non-persisting engine.
  persistView?: (view: CustomView) => Promise<void>;
  deleteView?: (id: string) => Promise<void>;
};

/**
 * Loads + persists/deletes the experiment's saved custom views, one experiment tag per
 * view (`mlflow.customView.view.v1.<id>`). Returns a no-op-free persist
 * callback only when an experiment id is present; without one the caller's
 * session-local fallback engages.
 */
export const useExperimentCustomViewDefinition = (experimentId?: string): ExperimentCustomViewDefinition => {
  const queryClient = useQueryClient();
  const queryKey = useMemo(() => ['experiment-custom-views', experimentId], [experimentId]);

  const { data, isLoading } = useQuery({
    queryKey,
    enabled: Boolean(experimentId),
    queryFn: async (): Promise<CustomView[]> => {
      const response = await MlflowService.getExperiment({ experiment_id: experimentId });
      const tags: ExperimentTag[] = response?.experiment?.tags ?? [];
      const customViewTags = tags.filter((tag) => tag.key.startsWith(CUSTOM_VIEW_TAG_PREFIX));
      // v1 is the only format today. When another version is introduced, dispatch each tag to its
      // version-specific parser while preserving one view (or unreadable placeholder) per tag so
      // views.length remains the version-agnostic quota count.
      const parsed = await Promise.all(
        customViewTags.map((tag) => deserializeView(tag.key.slice(CUSTOM_VIEW_PREFIX_V1.length), tag.value)),
      );
      return parsed.sort((a, b) => a.createdAtMs - b.createdAtMs);
    },
  });

  const persistMutation = useMutation({
    mutationFn: async (view: CustomView) => {
      const value = await serializeViewForTag(view);
      await MlflowService.setExperimentTag({ experiment_id: experimentId, key: viewTagKey(view.id), value });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  });

  const persistView = useCallback(
    async (view: CustomView) => {
      await persistMutation.mutateAsync(view);
    },
    [persistMutation],
  );

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      await MlflowService.deleteExperimentTag({ experiment_id: experimentId, key: viewTagKey(id) });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey }),
  });

  const deleteView = useCallback(
    async (id: string) => {
      await deleteMutation.mutateAsync(id);
    },
    [deleteMutation],
  );

  return {
    views: data ?? [],
    isLoaded: experimentId ? !isLoading : true,
    persistView: experimentId ? persistView : undefined,
    deleteView: experimentId ? deleteView : undefined,
  };
};
