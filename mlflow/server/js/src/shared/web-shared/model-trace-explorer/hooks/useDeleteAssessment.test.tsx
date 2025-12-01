import { describe, jest, beforeAll, beforeEach, it, expect, afterEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClientProvider, QueryClient } from '@databricks/web-shared/query-client';

import { useDeleteAssessment } from './useDeleteAssessment';
import { useTraceCachedActions } from './useTraceCachedActions';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment, ModelTraceInfoV3, ModelTraceLocation } from '../ModelTrace.types';
import { ModelTraceExplorerUpdateTraceContextProvider } from '../contexts/UpdateTraceContext';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: jest.fn(),
}));

describe('useDeleteAssessment', () => {
  const server = setupServer();
  beforeAll(() => server.listen());
  afterEach(() => server.resetHandlers());

  const mockTraceLocation: ModelTraceLocation = {
    type: 'UC_SCHEMA',
    uc_schema: {
      catalog_name: 'my_catalog',
      schema_name: 'my_schema',
    },
  };

  const mockTraceInfo: ModelTraceInfoV3 = {
    trace_id: 'trace-123',
    trace_location: mockTraceLocation,
    request_time: '2025-02-19T09:52:23.140Z',
    state: 'OK',
    tags: {},
  };

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <ModelTraceExplorerUpdateTraceContextProvider modelTraceInfo={mockTraceInfo} sqlWarehouseId="warehouse-123">
            {children}
          </ModelTraceExplorerUpdateTraceContextProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );

  const mockAssessment: Assessment = {
    assessment_id: 'assessment-123',
    assessment_name: 'Correctness',
    trace_id: 'trace-123',
    source: {
      source_type: 'HUMAN',
      source_id: 'user@databricks.com',
    },
    feedback: {
      value: 'yes',
    },
    create_time: '2025-11-16T12:00:00.000Z',
    last_update_time: '2025-11-16T12:00:00.000Z',
    valid: true,
  } as Assessment;

  const mockOverriddenAssessment: Assessment = {
    ...mockAssessment,
    assessment_id: 'assessment-old-llm',
    source: {
      source_type: 'LLM_JUDGE',
      source_id: 'databricks',
    },
    feedback: {
      value: 'no',
    },
    create_time: '2025-11-16T10:00:00.000Z',
    valid: false,
  } as Assessment;

  const mockAssessmentWithOverride: Assessment = {
    ...mockAssessment,
    overrides: 'assessment-old-llm',
    overriddenAssessment: mockOverriddenAssessment,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    renderHook(() => useTraceCachedActions()).result.current.resetCache();
  });

  describe('when V4 API is disabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
      server.use(
        rest.delete('*/ajax-api/*/mlflow/traces/trace-123/assessments/assessment-123', async (req, res, ctx) =>
          res(ctx.json({})),
        ),
      );
    });

    it('should delete assessment using V3 API', async () => {
      const deleteSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.delete('*/ajax-api/*/mlflow/traces/trace-123/assessments/assessment-123', async (req, res, ctx) => {
          deleteSpy();
          return res(ctx.json({}));
        }),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useDeleteAssessment({ assessment: mockAssessment, onSuccess }), { wrapper });

      result.current.deleteAssessmentMutation();

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(deleteSpy).toHaveBeenCalled();
    });

    it('should not update cache when V4 API is disabled', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          deleteHook: useDeleteAssessment({ assessment: mockAssessment, onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.deleteHook.deleteAssessmentMutation();

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(result.current.cacheHook.assessmentActions['trace-123']).toBeUndefined();
    });
  });

  describe('when V4 API is enabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(true);
      server.use(
        rest.delete(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) => res(ctx.json({})),
        ),
      );
    });

    it('should delete assessment using V4 API', async () => {
      const deleteSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.delete(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) => {
            deleteSpy();
            return res(ctx.json({}));
          },
        ),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useDeleteAssessment({ assessment: mockAssessment, onSuccess }), { wrapper });

      result.current.deleteAssessmentMutation();

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(deleteSpy).toHaveBeenCalled();
    });

    it('should add delete action to cache', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          deleteHook: useDeleteAssessment({ assessment: mockAssessment, onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.deleteHook.deleteAssessmentMutation();

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());

      const actions = result.current.cacheHook.assessmentActions['trace-123'];
      expect(actions).toHaveLength(1);
      expect(actions[0]).toMatchObject({
        action: 'DELETE',
        assessment: expect.objectContaining({ assessment_id: 'assessment-123' }),
      });
    });

    it('should restore overridden assessment when deleting an override', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          deleteHook: useDeleteAssessment({ assessment: mockAssessmentWithOverride, onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.deleteHook.deleteAssessmentMutation();

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());

      const actions = result.current.cacheHook.assessmentActions['trace-123'];
      expect(actions).toHaveLength(2);
      expect(actions[0]).toMatchObject({
        action: 'ADD',
        assessment: expect.objectContaining({
          assessment_id: 'assessment-old-llm',
          valid: true,
        }),
      });
      expect(actions[1]).toMatchObject({
        action: 'DELETE',
        assessment: expect.objectContaining({ assessment_id: 'assessment-123' }),
      });
    });

    it('should not restore assessment if there is no overriddenAssessment', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          deleteHook: useDeleteAssessment({ assessment: mockAssessment, onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.deleteHook.deleteAssessmentMutation();

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());

      const actions = result.current.cacheHook.assessmentActions['trace-123'];
      expect(actions).toHaveLength(1);
      expect(actions[0].action).toBe('DELETE');
    });

    it('should call onSettled callback', async () => {
      const onSettled = jest.fn();
      const { result } = renderHook(() => useDeleteAssessment({ assessment: mockAssessment, onSettled }), { wrapper });

      result.current.deleteAssessmentMutation();

      await waitFor(() => expect(onSettled).toHaveBeenCalled());
    });

    it('should not delete when skip is true', async () => {
      const deleteSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.delete(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) => {
            deleteSpy();
            return res(ctx.json({}));
          },
        ),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useDeleteAssessment({ assessment: mockAssessment, skip: true, onSuccess }), {
        wrapper,
      });

      result.current.deleteAssessmentMutation();

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(deleteSpy).not.toHaveBeenCalled();
      expect(onSuccess).not.toHaveBeenCalled();
    });
  });
});
