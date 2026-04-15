import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';

import { QueryClientProvider, QueryClient } from '../../query-client/queryClient';

import { useUpdateTraceTagsMutation } from './useUpdateTraceTagsMutation';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { ModelTraceInfoV3, ModelTraceLocation } from '../ModelTrace.types';
import { TracesServiceV3, TracesServiceV4 } from '../api';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: jest.fn(),
}));

jest.mock('../api', () => ({
  TracesServiceV3: {
    setTraceTagV3: jest.fn(),
    deleteTraceTagV3: jest.fn(),
  },
  TracesServiceV4: {
    setTraceTagV4: jest.fn(),
    deleteTraceTagV4: jest.fn(),
  },
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
    // Mock services to resolve successfully by default
    jest.mocked(TracesServiceV3.setTraceTagV3).mockResolvedValue({});
    jest.mocked(TracesServiceV3.deleteTraceTagV3).mockResolvedValue({});
    jest.mocked(TracesServiceV4.setTraceTagV4).mockResolvedValue({});
    jest.mocked(TracesServiceV4.deleteTraceTagV4).mockResolvedValue({});
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
        expect(TracesServiceV4.setTraceTagV4).toHaveBeenCalledTimes(1);
        expect(TracesServiceV4.deleteTraceTagV4).toHaveBeenCalledTimes(1);
        // Check call for new tag
        expect(TracesServiceV4.setTraceTagV4).toHaveBeenCalledWith({
          tag: { key: 'tag1', value: 'value1' },
          traceLocation: ucSchemaLocation,
          traceId: 'trace-456',
        });
        // Check call for deleted tag
        expect(TracesServiceV4.deleteTraceTagV4).toHaveBeenCalledWith({
          tagKey: 'tag2',
          traceLocation: ucSchemaLocation,
          traceId: 'trace-456',
        });
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
        expect(TracesServiceV3.setTraceTagV3).toHaveBeenCalledTimes(2);
        expect(TracesServiceV3.deleteTraceTagV3).toHaveBeenCalledTimes(2);
      });

      // Verify calls for new tags
      expect(TracesServiceV3.setTraceTagV3).toHaveBeenCalledWith({
        tag: { key: 'tag1', value: 'value1' },
        traceId: 'trace-456',
      });
      expect(TracesServiceV3.setTraceTagV3).toHaveBeenCalledWith({
        tag: { key: 'tag2', value: 'value2' },
        traceId: 'trace-456',
      });

      // Verify calls for deleted tags
      expect(TracesServiceV3.deleteTraceTagV3).toHaveBeenCalledWith({
        tagKey: 'tag3',
        traceId: 'trace-456',
      });
      expect(TracesServiceV3.deleteTraceTagV3).toHaveBeenCalledWith({
        tagKey: 'tag4',
        traceId: 'trace-456',
      });
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
        expect(TracesServiceV3.setTraceTagV3).toHaveBeenCalledTimes(1);
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
      expect(TracesServiceV3.setTraceTagV3).not.toHaveBeenCalled();
      expect(TracesServiceV3.deleteTraceTagV3).not.toHaveBeenCalled();
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
        expect(TracesServiceV3.setTraceTagV3).toHaveBeenCalledTimes(1);
        expect(TracesServiceV3.setTraceTagV3).toHaveBeenCalledWith({
          tag: { key: 'tag1', value: 'value1' },
          traceId: 'trace-456',
        });
        expect(TracesServiceV3.deleteTraceTagV3).not.toHaveBeenCalled();
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
        expect(TracesServiceV3.deleteTraceTagV3).toHaveBeenCalledTimes(1);
        expect(TracesServiceV3.deleteTraceTagV3).toHaveBeenCalledWith({
          tagKey: 'tag1',
          traceId: 'trace-456',
        });
        expect(TracesServiceV3.setTraceTagV3).not.toHaveBeenCalled();
      });
    });
  });
});
