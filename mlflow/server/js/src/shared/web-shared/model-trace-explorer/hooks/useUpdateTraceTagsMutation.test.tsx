import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';

import { QueryClientProvider, QueryClient } from '@databricks/web-shared/query-client';

import { useUpdateTraceTagsMutation } from './useUpdateTraceTagsMutation';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { ModelTraceInfoV3, ModelTraceLocation } from '../ModelTrace.types';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: jest.fn(),
}));

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: jest.fn((url) => url),
}));

describe('useUpdateTraceTagsMutation', () => {
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
    // Mock fetchAPI to resolve successfully by default
    jest.mocked(fetchAPI).mockResolvedValue({});
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

      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [{ key: 'tag1', value: 'value1' }],
        deletedTags: [{ key: 'tag2', value: 'value2' }],
        modelTraceInfo: traceInfoWithUcSchema,
      });

      await waitFor(() => {
        expect(fetchAPI).toHaveBeenCalledTimes(2);
        // Check PATCH call for new tag
        expect(fetchAPI).toHaveBeenCalledWith(
          expect.stringContaining('ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-456/tags'),
          expect.objectContaining({ method: 'PATCH', body: { key: 'tag1', value: 'value1' } }),
        );
        // Check DELETE call for deleted tag
        expect(fetchAPI).toHaveBeenCalledWith(
          expect.stringContaining('ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-456/tags/tag2'),
          expect.objectContaining({ method: 'DELETE' }),
        );
      });
    });
  });

  describe('when V4 API is disabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    });

    it('should use V3 endpoints', async () => {
      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags,
        deletedTags,
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(fetchAPI).toHaveBeenCalledTimes(4);
      });

      // Verify PATCH calls for new tags
      expect(fetchAPI).toHaveBeenCalledWith(
        expect.stringContaining('ajax-api/3.0/mlflow/traces/trace-456/tags'),
        expect.objectContaining({ method: 'PATCH', body: { key: 'tag1', value: 'value1' } }),
      );
      expect(fetchAPI).toHaveBeenCalledWith(
        expect.stringContaining('ajax-api/3.0/mlflow/traces/trace-456/tags'),
        expect.objectContaining({ method: 'PATCH', body: { key: 'tag2', value: 'value2' } }),
      );

      // Verify DELETE calls for deleted tags
      expect(fetchAPI).toHaveBeenCalledWith(
        expect.stringMatching(/ajax-api\/3\.0\/mlflow\/traces\/trace-456\/tags.*key=tag3/),
        expect.objectContaining({ method: 'DELETE' }),
      );
      expect(fetchAPI).toHaveBeenCalledWith(
        expect.stringMatching(/ajax-api\/3\.0\/mlflow\/traces\/trace-456\/tags.*key=tag4/),
        expect.objectContaining({ method: 'DELETE' }),
      );
    });
  });

  describe('onSuccess callback', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    });

    it('should call onSuccess callback after successful mutation', async () => {
      const onSuccess = jest.fn();

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
      expect(fetchAPI).not.toHaveBeenCalled();
    });

    it('should handle only newTags', async () => {
      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [{ key: 'tag1', value: 'value1' }],
        deletedTags: [],
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(fetchAPI).toHaveBeenCalledTimes(1);
        expect(fetchAPI).toHaveBeenCalledWith(
          expect.stringContaining('ajax-api/3.0/mlflow/traces/trace-456/tags'),
          expect.objectContaining({ method: 'PATCH' }),
        );
      });
    });

    it('should handle only deletedTags', async () => {
      const { result } = renderHook(() => useUpdateTraceTagsMutation({}), {
        wrapper,
      });

      await result.current.mutateAsync({
        newTags: [],
        deletedTags: [{ key: 'tag1', value: 'value1' }],
        modelTraceInfo: mockTraceInfoV3,
      });

      await waitFor(() => {
        expect(fetchAPI).toHaveBeenCalledTimes(1);
        expect(fetchAPI).toHaveBeenCalledWith(
          expect.stringMatching(/ajax-api\/3\.0\/mlflow\/traces\/trace-456\/tags.*key=tag1/),
          expect.objectContaining({ method: 'DELETE' }),
        );
      });
    });
  });
});
