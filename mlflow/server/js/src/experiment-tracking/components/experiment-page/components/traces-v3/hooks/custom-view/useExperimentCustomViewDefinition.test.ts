import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { renderHook, waitFor } from '@testing-library/react';

import {
  type CustomView,
  CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES,
  getUtf8ByteLength,
  parseCustomView,
  serializeCustomView,
  viewTagKey,
} from '@databricks/web-shared/model-trace-explorer/custom-view';

import { useExperimentCustomViewDefinition } from './useExperimentCustomViewDefinition';
import { MlflowService } from '../../../../../../sdk/MlflowService';
import type { GetExperimentApiResponse } from '../../../../../../types';
import { isTextCompressedDeflate, textCompressDeflate } from '../../../../../../../common/utils/StringUtils';

// parseCustomView is spyable so one test can force it to throw (simulating a
// RangeError from validateTemplate's recursive walk over a hostile/deeply-nested
// tag). It delegates to the real implementation by default so every other test
// is unaffected.
jest.mock('@databricks/web-shared/model-trace-explorer/custom-view', () => {
  const actual = jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer/custom-view')>(
    '@databricks/web-shared/model-trace-explorer/custom-view',
  );
  return { __esModule: true, ...actual, parseCustomView: jest.fn(actual.parseCustomView) };
});

const EXPERIMENT_ID = 'exp-123';

// Builds a valid, correctly-typed CustomView by round-tripping a well-formed
// stored shape through parseCustomView (so the template is a real, validated
// A2uiMessage[] without needing a hand-typed literal or a cast). Overrides are
// applied on top for per-test fields like id / createdAtMs / instruction.
const makeView = (overrides: Partial<CustomView> = {}): CustomView => {
  const base = parseCustomView({
    id: 'view-1',
    name: 'My view',
    label: 'My view',
    instruction: 'show text',
    createdAtMs: 100,
    template: [
      {
        version: 'v0.9',
        updateComponents: {
          surfaceId: 'main',
          components: [{ id: 'root', component: 'Text', text: 'Hello' }],
        },
      },
    ],
  });
  if (!base) {
    throw new Error('test fixture failed to validate; the stored shape is no longer a valid CustomView');
  }
  return { ...base, ...overrides };
};

const customViewTag = (view: CustomView) => ({ key: viewTagKey(view.id), value: serializeCustomView(view) });

const getExperimentResponse = (tags: { key: string; value: string }[]): GetExperimentApiResponse => ({
  experiment: {
    experimentId: EXPERIMENT_ID,
    artifactLocation: 'dbfs:/tmp',
    creationTime: 0,
    lastUpdateTime: 0,
    lifecycleStage: 'active',
    name: 'exp',
    tags,
  },
});

// Deterministic high-entropy string: an xorshift PRNG mapped onto printable
// ASCII. Deflate cannot meaningfully shrink this, so a large one exceeds the tag
// cap even after compression — driving the too-large hard-fail path.
const makeIncompressibleString = (length: number): string => {
  let seed = 0x9e3779b9;
  let out = '';
  for (let i = 0; i < length; i++) {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    seed >>>= 0;
    out += String.fromCharCode(33 + (seed % 94));
  }
  return out;
};

const renderDefinition = (queryClient: QueryClient, experimentId?: string) =>
  renderHook(() => useExperimentCustomViewDefinition(experimentId), {
    // React.createElement (not JSX) keeps this a `.test.ts` file — JSX would
    // require a `.tsx` extension.
    wrapper: ({ children }) => React.createElement(QueryClientProvider, { client: queryClient }, children),
  });

describe('useExperimentCustomViewDefinition', () => {
  let queryClient: QueryClient;
  let mockGetExperiment: jest.SpiedFunction<typeof MlflowService.getExperiment>;
  let mockSetExperimentTag: jest.SpiedFunction<typeof MlflowService.setExperimentTag>;
  let mockDeleteExperimentTag: jest.SpiedFunction<typeof MlflowService.deleteExperimentTag>;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    jest.clearAllMocks();
    mockGetExperiment = jest.spyOn(MlflowService, 'getExperiment').mockResolvedValue(getExperimentResponse([]));
    mockSetExperimentTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
    mockDeleteExperimentTag = jest.spyOn(MlflowService, 'deleteExperimentTag').mockResolvedValue({});
  });

  describe('load path', () => {
    it('loads all custom-view tags, keeps corrupt values as unreadable placeholders, and sorts by createdAtMs', async () => {
      const newer = makeView({ id: 'newer', createdAtMs: 200 });
      const older = makeView({ id: 'older', createdAtMs: 100 });

      mockGetExperiment.mockResolvedValue(
        getExperimentResponse([
          // Presented newest-first so the sort has something to reorder.
          customViewTag(newer),
          customViewTag(older),
          // Empty values are not tombstones in OSS because deletion hard-deletes the tag. Treat a
          // directly-created empty tag like any other corrupt value so it still consumes a slot.
          { key: viewTagKey('empty'), value: '' },
          // Corrupt payloads under the custom-view prefix → kept as `unreadable`
          // placeholders (keyed by the recovered id) rather than dropped, so an
          // incompatible saved view stays selectable instead of vanishing.
          { key: viewTagKey('bad-json'), value: '{not valid json' },
          { key: viewTagKey('wrong-shape'), value: JSON.stringify({ foo: 'bar' }) },
          // Shape mismatch but carries a string `name` → salvage it for display.
          { key: viewTagKey('named-bad'), value: JSON.stringify({ name: 'Salvaged name' }) },
          // Unrelated experiment tag (no custom-view prefix) → filtered out.
          { key: 'mlflow.note.content', value: 'a note' },
        ]),
      );

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);

      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      expect(mockGetExperiment).toHaveBeenCalledWith({ experiment_id: EXPERIMENT_ID });
      // Unreadable placeholders get createdAtMs 0, so they sort ahead of the valid views; only the
      // non-prefixed note is excluded.
      expect(result.current.views.map((view) => view.id)).toEqual([
        'empty',
        'bad-json',
        'wrong-shape',
        'named-bad',
        'older',
        'newer',
      ]);
      const byId = (id: string) => result.current.views.find((view) => view.id === id);
      // Unparseable bytes and shape mismatches are kept and flagged unreadable.
      expect(byId('empty')?.unreadable).toBe(true);
      expect(byId('bad-json')?.unreadable).toBe(true);
      expect(byId('wrong-shape')?.unreadable).toBe(true);
      expect(byId('named-bad')?.unreadable).toBe(true);
      // No salvageable name → empty (the render layer supplies the translated
      // "Untitled custom view" fallback); a string `name` is salvaged for display.
      expect(byId('bad-json')?.name).toBe('');
      expect(byId('named-bad')?.name).toBe('Salvaged name');
      // Valid views are untouched (not flagged unreadable).
      expect(byId('older')?.unreadable).toBeUndefined();
      expect(byId('newer')?.unreadable).toBeUndefined();
    });

    it('keeps a valid-shape view whose template fails validation as a normal (not unreadable) view', async () => {
      // Valid CustomView shape (string id, array template) but the template has no
      // `root` component, so validateTemplate would reject it. Load no longer
      // validates templates (that walk is deferred to the selection/render path),
      // so it loads as a normal view with its template preserved and `unreadable`
      // unset; the render-time gate shows the placeholder when it's selected.
      const invalidTemplate = [
        {
          version: 'v0.9',
          updateComponents: { surfaceId: 'main', components: [{ id: 'a', component: 'Text', text: 'x' }] },
        },
      ];
      mockGetExperiment.mockResolvedValue(
        getExperimentResponse([
          {
            key: viewTagKey('bad-template'),
            value: JSON.stringify({
              id: 'bad-template',
              name: 'Bad template',
              createdAtMs: 10,
              template: invalidTemplate,
            }),
          },
        ]),
      );

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const view = result.current.views.find((v) => v.id === 'bad-template');
      expect(view?.unreadable).toBeUndefined();
      expect(view?.name).toBe('Bad template');
      expect(view?.template).toEqual(invalidTemplate);
    });

    it('decompresses deflate-compressed tag values on read', async () => {
      const view = makeView({ id: 'compressed', createdAtMs: 5 });
      const compressedValue = await textCompressDeflate(serializeCustomView(view));
      expect(isTextCompressedDeflate(compressedValue)).toBe(true);

      mockGetExperiment.mockResolvedValue(
        getExperimentResponse([{ key: viewTagKey(view.id), value: compressedValue }]),
      );

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);

      await waitFor(() => expect(result.current.views).toHaveLength(1));
      expect(result.current.views[0]).toEqual(view);
    });

    it('contains a throw inside parseCustomView to a single unreadable placeholder without dropping other views', async () => {
      const valid = makeView({ id: 'valid', createdAtMs: 50 });

      const actual = jest.requireActual<typeof import('@databricks/web-shared/model-trace-explorer/custom-view')>(
        '@databricks/web-shared/model-trace-explorer/custom-view',
      );
      const mockParse = jest.mocked(parseCustomView);
      // Simulate validateTemplate's recursive walk blowing the stack on a
      // hostile/deeply-nested tag: throw for the sentinel payload, delegate the
      // rest to the real parser.
      mockParse.mockImplementation((value: unknown) => {
        if (value && typeof value === 'object' && (value as { __boom?: unknown }).__boom) {
          throw new RangeError('Maximum call stack size exceeded');
        }
        return actual.parseCustomView(value);
      });

      try {
        mockGetExperiment.mockResolvedValue(
          getExperimentResponse([
            customViewTag(valid),
            { key: viewTagKey('boom'), value: JSON.stringify({ id: 'boom', name: 'Boom', __boom: true }) },
          ]),
        );

        const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
        await waitFor(() => expect(result.current.isLoaded).toBe(true));

        // The throw is contained: the valid view still loads and the throwing tag
        // degrades to one unreadable placeholder (name salvaged) instead of
        // rejecting Promise.all and dropping every saved view.
        expect(result.current.views.map((view) => view.id)).toEqual(['boom', 'valid']);
        const boom = result.current.views.find((view) => view.id === 'boom');
        expect(boom?.unreadable).toBe(true);
        expect(boom?.name).toBe('Boom');
      } finally {
        // Restore delegation so the override does not leak into later tests
        // (clearAllMocks resets call data but not implementations).
        mockParse.mockImplementation(actual.parseCustomView);
      }
    });

    it('keys a readable view by its tag-key suffix, not the (possibly divergent) payload id', async () => {
      const view = makeView({ id: 'payload-id', name: 'Diverged', createdAtMs: 10 });

      // Store the view under a tag key whose suffix does NOT match the serialized
      // payload id (only reachable via manual/external tag mutation). The loader
      // must treat the tag-key suffix as authoritative so save addresses the
      // original tag rather than the stale payload id.
      mockGetExperiment.mockResolvedValue(
        getExperimentResponse([{ key: viewTagKey('tag-key-id'), value: serializeCustomView(view) }]),
      );

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.views).toHaveLength(1));

      const loaded = result.current.views[0];
      expect(loaded.id).toBe('tag-key-id');
      // The rest of the readable view is preserved untouched.
      expect(loaded.unreadable).toBeUndefined();
      expect(loaded.name).toBe('Diverged');
    });
  });

  describe('persist path', () => {
    it('stores small views uncompressed and invalidates the cached views', async () => {
      const invalidateQueriesSpy = jest.spyOn(queryClient, 'invalidateQueries');
      const view = makeView({ id: 'small' });

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const persistView = result.current.persistView;
      if (!persistView) {
        throw new Error('persistView should be defined when an experiment id is present');
      }
      await persistView(view);

      const call = mockSetExperimentTag.mock.calls[0][0];
      expect(call).toEqual({
        experiment_id: EXPERIMENT_ID,
        key: viewTagKey('small'),
        value: serializeCustomView(view),
      });
      expect(isTextCompressedDeflate(call.value)).toBe(false);
      expect(invalidateQueriesSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ['experiment-custom-views', EXPERIMENT_ID] }),
      );
    });

    it('compresses views whose raw JSON exceeds the safe client limit', async () => {
      const view = makeView({ id: 'large', instruction: 'a'.repeat(70000) });
      expect(getUtf8ByteLength(serializeCustomView(view))).toBeGreaterThan(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES);

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const persistView = result.current.persistView;
      if (!persistView) {
        throw new Error('persistView should be defined when an experiment id is present');
      }
      await persistView(view);

      const call = mockSetExperimentTag.mock.calls[0][0];
      expect(call.key).toBe(viewTagKey('large'));
      expect(isTextCompressedDeflate(call.value)).toBe(true);
      expect(getUtf8ByteLength(call.value)).toBeLessThanOrEqual(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES);
    });

    it('uses UTF-8 bytes to decide whether to compress', async () => {
      const view = makeView({ id: 'unicode', instruction: '🙂'.repeat(6000) });
      const raw = serializeCustomView(view);
      expect(raw.length).toBeLessThan(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES);
      expect(getUtf8ByteLength(raw)).toBeGreaterThan(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES);

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const persistView = result.current.persistView;
      if (!persistView) {
        throw new Error('persistView should be defined when an experiment id is present');
      }
      await persistView(view);

      const call = mockSetExperimentTag.mock.calls[0][0];
      expect(isTextCompressedDeflate(call.value)).toBe(true);
    });

    it('stores raw JSON at the 20,000-byte safe client limit', async () => {
      expect(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES).toBe(20000);
      const baseView = makeView({ id: 'raw-boundary', instruction: '' });
      const instructionLength = CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES - getUtf8ByteLength(serializeCustomView(baseView));
      expect(instructionLength).toBeGreaterThanOrEqual(0);
      const view = { ...baseView, instruction: 'a'.repeat(instructionLength) };
      const raw = serializeCustomView(view);
      expect(getUtf8ByteLength(raw)).toBe(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES);

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const persistView = result.current.persistView;
      if (!persistView) {
        throw new Error('persistView should be defined when an experiment id is present');
      }
      await persistView(view);

      expect(mockSetExperimentTag).toHaveBeenCalledWith({
        experiment_id: EXPERIMENT_ID,
        key: viewTagKey('raw-boundary'),
        value: raw,
      });
    });

    it('hard-fails when even the compressed view exceeds the safe client limit', async () => {
      const view = makeView({ id: 'too-large', instruction: makeIncompressibleString(100000) });
      const compressed = await textCompressDeflate(serializeCustomView(view));
      expect(getUtf8ByteLength(compressed)).toBeGreaterThan(CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES);

      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const persistView = result.current.persistView;
      if (!persistView) {
        throw new Error('persistView should be defined when an experiment id is present');
      }

      await expect(persistView(view)).rejects.toThrow('too large to save');
      expect(mockSetExperimentTag).not.toHaveBeenCalled();
    });
  });

  describe('delete path', () => {
    it('hard deletes the saved-view tag and invalidates the cached views', async () => {
      const invalidateQueriesSpy = jest.spyOn(queryClient, 'invalidateQueries');
      const { result } = renderDefinition(queryClient, EXPERIMENT_ID);
      await waitFor(() => expect(result.current.isLoaded).toBe(true));

      const deleteView = result.current.deleteView;
      if (!deleteView) {
        throw new Error('deleteView should be defined when an experiment id is present');
      }
      await deleteView('view-1');

      expect(mockDeleteExperimentTag).toHaveBeenCalledWith({
        experiment_id: EXPERIMENT_ID,
        key: viewTagKey('view-1'),
      });
      expect(invalidateQueriesSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ['experiment-custom-views', EXPERIMENT_ID] }),
      );
    });
  });

  describe('without an experiment id', () => {
    it('reports loaded, exposes no mutation callbacks, and never fetches', () => {
      const { result } = renderDefinition(queryClient, undefined);

      expect(result.current.isLoaded).toBe(true);
      expect(result.current.views).toEqual([]);
      expect(result.current.persistView).toBeUndefined();
      expect(result.current.deleteView).toBeUndefined();
      expect(mockGetExperiment).not.toHaveBeenCalled();
      expect(mockDeleteExperimentTag).not.toHaveBeenCalled();
    });
  });
});
