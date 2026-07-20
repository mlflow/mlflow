import { describe, it, expect } from '@jest/globals';
import { textCompressDeflate, isTextCompressedDeflate } from '../../../../common/utils/StringUtils';
import { EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX } from '../../../constants';
import {
  decodeSavedViewEnvelope,
  deserializePersistedState,
  encodeSavedViewEnvelope,
  getSavedViewIdFromTagKey,
  getSavedViewTagKey,
  listSavedViews,
} from './savedViewEnvelope';

describe('savedViewEnvelope', () => {
  describe('tag key <-> id', () => {
    it('builds a prefixed tag key from an id', () => {
      expect(getSavedViewTagKey('abc123')).toEqual(`${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}abc123`);
    });

    it('extracts the id from a prefixed tag key', () => {
      expect(getSavedViewIdFromTagKey(`${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}abc123`)).toEqual('abc123');
    });

    it('returns null for a key without the saved-view prefix', () => {
      expect(getSavedViewIdFromTagKey('mlflow.note.content')).toBeNull();
    });

    it('returns null when the key is exactly the prefix with no id', () => {
      expect(getSavedViewIdFromTagKey(EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX)).toBeNull();
    });
  });

  describe('encode', () => {
    it('produces JSON with plain-text name/createdAt and leaves the state blob untouched', async () => {
      const compressedState = await textCompressDeflate(JSON.stringify({ selectedColumns: ['accuracy'] }));
      const envelopeJson = encodeSavedViewEnvelope('My view', compressedState, 1770000000000);

      // name and createdAt must be readable without inflating anything
      const parsed = JSON.parse(envelopeJson);
      expect(parsed.name).toEqual('My view');
      expect(parsed.createdAt).toEqual(1770000000000);
      expect(parsed.state).toEqual(compressedState);
      expect(isTextCompressedDeflate(parsed.state)).toBe(true);
    });
  });

  describe('decode', () => {
    it('round-trips an encoded envelope', async () => {
      const compressedState = await textCompressDeflate(JSON.stringify({ selectedColumns: ['accuracy'] }));
      const envelopeJson = encodeSavedViewEnvelope('My view', compressedState, 1770000000000);

      const decoded = decodeSavedViewEnvelope(envelopeJson);
      expect(decoded).toEqual({ name: 'My view', createdAt: 1770000000000, state: compressedState });
    });

    it('throws on malformed JSON', () => {
      expect(() => decodeSavedViewEnvelope('not json {')).toThrow();
    });

    it('throws when required fields are missing', () => {
      expect(() => decodeSavedViewEnvelope(JSON.stringify({ name: 'no state' }))).toThrow();
    });
  });

  describe('deserializePersistedState', () => {
    it('deserializes the compressed state field back into the original object', async () => {
      const original = { selectedColumns: ['accuracy'], groupBy: 'foo' };
      const compressedState = await textCompressDeflate(JSON.stringify(original));
      const envelope = decodeSavedViewEnvelope(encodeSavedViewEnvelope('v', compressedState, 1));

      expect(await deserializePersistedState(envelope)).toEqual(original);
    });

    it('supports an uncompressed (plain JSON) state field for forward-compat', async () => {
      const original = { selectedColumns: ['loss'] };
      const envelope = { name: 'v', createdAt: 1, state: JSON.stringify(original) };

      expect(await deserializePersistedState(envelope)).toEqual(original);
    });
  });

  describe('listSavedViews', () => {
    const buildTag = async (id: string, name: string, createdAt: number) => ({
      key: getSavedViewTagKey(id),
      value: encodeSavedViewEnvelope(name, await textCompressDeflate('{}'), createdAt),
    });

    it('filters experiment tags by the saved-view prefix and returns summaries without inflating state', async () => {
      const tags = [
        { key: 'mlflow.note.content', value: 'a note' },
        await buildTag('v1', 'First view', 300),
        await buildTag('v2', 'Second view', 100),
      ];

      const views = listSavedViews(tags);

      expect(views).toEqual([
        { id: 'v1', name: 'First view', createdAt: 300 },
        { id: 'v2', name: 'Second view', createdAt: 100 },
      ]);
    });

    it('skips tags whose value fails to decode rather than throwing', async () => {
      const tags = [{ key: getSavedViewTagKey('bad'), value: 'not-json{' }, await buildTag('good', 'Good view', 5)];

      expect(listSavedViews(tags)).toEqual([{ id: 'good', name: 'Good view', createdAt: 5 }]);
    });

    it('returns an empty array when there are no saved-view tags', () => {
      expect(listSavedViews([{ key: 'mlflow.user', value: 'someone' }])).toEqual([]);
    });

    it('skips a bare-prefix tag with no id rather than emitting an empty-id summary', async () => {
      const tags = [
        {
          key: EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX,
          value: encodeSavedViewEnvelope('No id', await textCompressDeflate('{}'), 1),
        },
        await buildTag('good', 'Good view', 5),
      ];

      expect(listSavedViews(tags)).toEqual([{ id: 'good', name: 'Good view', createdAt: 5 }]);
    });
  });
});
