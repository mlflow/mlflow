import type { KeyValueEntity } from '../../../../common/types';
import { isTextCompressedDeflate, textDecompressDeflate } from '../../../../common/utils/StringUtils';
import { EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX } from '../../../constants';

/**
 * A saved view is stored as a single experiment tag: the key is the share prefix plus a stable
 * opaque id, and the value is a JSON "envelope". The envelope keeps `name` and `createdAt` as plain
 * JSON so the views list can be rendered without deserializing anything; only `state` (the
 * serialized ExperimentPageUIState + search facets, the same payload the share link embeds) is
 * deflate-compressed, and it is deserialized lazily at apply-time for the single view the user opens.
 */
export interface SavedViewEnvelope {
  name: string;
  createdAt: number;
  // When the view was last overwritten in place. Absent on views written before overwrite existed
  // (and by callers that never overwrite, e.g. runs / V3); decode defaults it to `createdAt`.
  updatedAt: number;
  // The compressed (or, for forward-compat, plain-JSON) serialized view state.
  state: string;
}

/**
 * Lightweight summary used to render the views list without deserializing any state.
 */
export interface SavedViewSummary {
  id: string;
  name: string;
  createdAt: number;
  // Present for callers that overwrite views (traces V4); the runs / V3 lists ignore it.
  updatedAt?: number;
}

export const getSavedViewTagKey = (id: string): string => `${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}${id}`;

export const getSavedViewIdFromTagKey = (tagKey: string): string | null => {
  if (!tagKey.startsWith(EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX)) {
    return null;
  }
  // A key that is exactly the prefix (no id) yields an empty id, which would collide across tags;
  // treat it as not a saved-view key.
  const id = tagKey.slice(EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX.length);
  return id === '' ? null : id;
};

/**
 * Build the experiment-tag value for a saved view. `compressedState` is the already-serialized
 * (typically deflate-compressed) view-state blob; it is stored verbatim so the name/createdAt stay
 * readable without decompression.
 */
// `updatedAt` is optional so callers that never overwrite (runs / V3) can keep the 3-arg form; it
// defaults to `createdAt` for a freshly-created view.
export const encodeSavedViewEnvelope = (
  name: string,
  compressedState: string,
  createdAt: number,
  updatedAt: number = createdAt,
): string => JSON.stringify({ name, createdAt, updatedAt, state: compressedState } satisfies SavedViewEnvelope);

// `updatedAt` is validated loosely (may be absent on legacy tags); decode fills it from `createdAt`.
const isValidEnvelope = (value: unknown): value is Omit<SavedViewEnvelope, 'updatedAt'> & { updatedAt?: unknown } =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as SavedViewEnvelope).name === 'string' &&
  typeof (value as SavedViewEnvelope).createdAt === 'number' &&
  typeof (value as SavedViewEnvelope).state === 'string';

/**
 * Parse an experiment-tag value into a saved-view envelope. Throws if the value is not valid JSON
 * or is missing required fields; the `state` field is left compressed (deserialize lazily via
 * {@link deserializePersistedState}). A missing/invalid `updatedAt` falls back to `createdAt`, so
 * views written before overwrite existed decode cleanly.
 */
export const decodeSavedViewEnvelope = (tagValue: string): SavedViewEnvelope => {
  const parsed = JSON.parse(tagValue);
  if (!isValidEnvelope(parsed)) {
    throw new Error(
      'Invalid saved-view envelope: expected an object with a string `name`, number `createdAt`, and string `state`',
    );
  }
  const updatedAt = typeof parsed.updatedAt === 'number' ? parsed.updatedAt : parsed.createdAt;
  return { name: parsed.name, createdAt: parsed.createdAt, updatedAt, state: parsed.state };
};

/**
 * Deserialize an envelope's `state` field back into the serialized view-state object. Supports both
 * a deflate-compressed blob and a plain-JSON string (forward-compat with uncompressed writes).
 */
export const deserializePersistedState = async (envelope: SavedViewEnvelope): Promise<unknown> => {
  const raw = isTextCompressedDeflate(envelope.state) ? await textDecompressDeflate(envelope.state) : envelope.state;
  return JSON.parse(raw);
};

/**
 * Normalize the `experimentTagsByExperimentId` slice into plain {key, value} entities. The reducer
 * stores Immutable ExperimentTag records (getKey/getValue) in production, but fixtures and some
 * mock stores hold plain objects, so read either shape rather than assuming the Immutable API.
 */
export const toKeyValueEntities = (tagsById: unknown): KeyValueEntity[] =>
  Object.values((tagsById as Record<string, any>) ?? {}).map((tag: any) => ({
    key: typeof tag?.getKey === 'function' ? tag.getKey() : tag.key,
    value: typeof tag?.getValue === 'function' ? tag.getValue() : tag.value,
  }));

/**
 * Filter a set of experiment tags down to saved-view summaries, in tag order. Tags whose value
 * fails to decode are skipped rather than throwing, so one corrupt view can't break the whole list.
 * State is intentionally not deserialized here — the list only needs the name and createdAt.
 */
export const listSavedViews = (tags: KeyValueEntity[]): SavedViewSummary[] =>
  tags.reduce<SavedViewSummary[]>((views, { key, value }) => {
    const id = getSavedViewIdFromTagKey(key);
    if (id === null) {
      return views;
    }
    try {
      const { name, createdAt, updatedAt } = decodeSavedViewEnvelope(value);
      views.push({ id, name, createdAt, updatedAt });
    } catch {
      // Skip a corrupt/legacy tag value rather than failing the entire list.
    }
    return views;
  }, []);
