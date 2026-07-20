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
export const encodeSavedViewEnvelope = (name: string, compressedState: string, createdAt: number): string =>
  JSON.stringify({ name, createdAt, state: compressedState } satisfies SavedViewEnvelope);

const isValidEnvelope = (value: unknown): value is SavedViewEnvelope =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as SavedViewEnvelope).name === 'string' &&
  typeof (value as SavedViewEnvelope).createdAt === 'number' &&
  typeof (value as SavedViewEnvelope).state === 'string';

/**
 * Parse an experiment-tag value into a saved-view envelope. Throws if the value is not valid JSON
 * or is missing required fields; the `state` field is left compressed (deserialize lazily via
 * {@link deserializePersistedState}).
 */
export const decodeSavedViewEnvelope = (tagValue: string): SavedViewEnvelope => {
  const parsed = JSON.parse(tagValue);
  if (!isValidEnvelope(parsed)) {
    throw new Error(
      'Invalid saved-view envelope: expected an object with a string `name`, number `createdAt`, and string `state`',
    );
  }
  return { name: parsed.name, createdAt: parsed.createdAt, state: parsed.state };
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
      const { name, createdAt } = decodeSavedViewEnvelope(value);
      views.push({ id, name, createdAt });
    } catch {
      // Skip a corrupt/legacy tag value rather than failing the entire list.
    }
    return views;
  }, []);
