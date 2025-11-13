import { describe, jest, beforeAll, beforeEach, it, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from 'msw/node';

import { QueryClientProvider, QueryClient } from '@databricks/web-shared/query-client';

import { useUpdateTraceTagsMutation } from './useUpdateTraceTagsMutation';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { ModelTraceInfoV3, ModelTraceLocation } from '../ModelTrace.types';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: jest.fn(),
}));

describe('useUpdateTraceTagsMutation', () => {
  const server = setupServer();
  beforeAll(() => server.listen());

  const wrapper = ({ children }: { children: React.ReactNode }) => {
    const queryClient = new QueryClient();
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };

  const mockTraceLocation: ModelTraceLocation = {
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: {
      experiment_id: 'exp-123',
    },
  };

  const mockTraceInfoV3: ModelTraceInfoV3 = {
    trace_id: 'trace-456',
    trace_location: mockTraceLocation,
    request_time: '2025-02-19T09:52:23.140Z',
    state: 'OK',
    tags: {},
  };

  const newTags = [
    { key: 'tag1', value: 'value1' },
    { key: 'tag2', value: 'value2' },
  ];

  const deletedTags = [
    { key: 'tag3', value: 'value3' },
    { key: 'tag4', value: 'value4' },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('when V4 API is enabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(true);
    });

    it('should use V4 endpoints for UC_SCHEMA location', async () => {
      const ucSchemaLocation: ModelTraceLocation = {
        type: 'UC_SCHEMA',
        uc_schema: {
          catalog_name: 'my_catalog',
          schema_name: 'my_schema',
        },
      };

      const traceInfoWithUcSchema: ModelTraceInfoV3 = {
        ...mockTraceInfoV3,
        trace_location: ucSchemaLocation,
      };

      const patchSpy = jest.fn();
      const deleteSpy = jest.fn();

      server.use(
        rest.patch('ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-456/tags', async (req, res, ctx) => {
          patchSpy(await req.json());
          return res(ctx.json({}));
        }),
        rest.delete('ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-456/tags/:tagKey', async (req, res, ctx) => {
          deleteSpy(req.params['tagKey']);
          return res(ctx.json({}));
        }),
      );

      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [{ key: 'tag1', value: 'value1' }],
        deletedTags: [{ key: 'tag2', value: 'value2' }],
        modelTraceInfo: traceInfoWithUcSchema,
      });

      await waitFor(() => {
        expect(patchSpy).toHaveBeenCalledWith({ key: 'tag1', value: 'value1' });
        expect(deleteSpy).toHaveBeenCalledWith('tag2');
      });
    });
  });

  describe('when V4 API is disabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    });

    it('should use V3 endpoints', async () => {
      const v3TagRequests: { method: string; body?: any; url: string }[] = [];

      server.use(
        rest.patch('http://localhost/ajax-api/3.0/mlflow/traces/trace-456/tags', async (req, res, ctx) => {
          v3TagRequests.push({ method: 'PATCH', body: await req.json(), url: req.url.href });
          return res(ctx.json({}));
        }),
        rest.delete('http://localhost/ajax-api/3.0/mlflow/traces/trace-456/tags', async (req, res, ctx) => {
          v3TagRequests.push({ method: 'DELETE', url: req.url.href });
          return res(ctx.json({}));
        }),
      );

      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags,
        deletedTags,
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(v3TagRequests).toHaveLength(4);
      });

      // Verify creation requests
      const createRequests = v3TagRequests.filter((req) => req.method === 'PATCH');
      expect(createRequests).toHaveLength(2);
      expect(createRequests[0].body).toEqual({ key: 'tag1', value: 'value1' });
      expect(createRequests[1].body).toEqual({ key: 'tag2', value: 'value2' });

      // Verify deletion requests
      const deleteRequests = v3TagRequests.filter((req) => req.method === 'DELETE');
      expect(deleteRequests).toHaveLength(2);
      expect(deleteRequests[0].url).toContain('key=tag3');
      expect(deleteRequests[1].url).toContain('key=tag4');
    });
  });

  describe('onSuccess callback', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    });

    it('should call onSuccess callback after successful mutation', async () => {
      const onSuccess = jest.fn();

      server.use(
        rest.patch('http://localhost/ajax-api/3.0/mlflow/traces/trace-456/tags', async (req, res, ctx) => {
          return res(ctx.json({}));
        }),
      );

      const { result } = renderHook(() => useUpdateTraceTagsMutation({ onSuccess }), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [{ key: 'tag1', value: 'value1' }],
        deletedTags: [],
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(onSuccess).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('edge cases', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    });

    it('should handle empty newTags and deletedTags arrays', async () => {
      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [],
        deletedTags: [],
        modelTraceInfo: mockTraceInfoV3,
      });

      // Should complete without making any requests
      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });
    });

    it('should handle only newTags', async () => {
      const v3TagRequests: { method: string }[] = [];

      server.use(
        rest.patch('http://localhost/ajax-api/3.0/mlflow/traces/trace-456/tags', async (req, res, ctx) => {
          v3TagRequests.push({ method: 'PATCH' });
          return res(ctx.json({}));
        }),
      );

      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [{ key: 'tag1', value: 'value1' }],
        deletedTags: [],
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(v3TagRequests).toHaveLength(1);
        expect(v3TagRequests[0].method).toBe('PATCH');
      });
    });

    it('should handle only deletedTags', async () => {
      const v3TagRequests: { method: string }[] = [];

      server.use(
        rest.delete('http://localhost/ajax-api/3.0/mlflow/traces/trace-456/tags', async (req, res, ctx) => {
          v3TagRequests.push({ method: 'DELETE' });
          return res(ctx.json({}));
        }),
      );

      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [],
        deletedTags: [{ key: 'tag1', value: 'value1' }],
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(v3TagRequests).toHaveLength(1);
        expect(v3TagRequests[0].method).toBe('DELETE');
      });
    });
  });
});
