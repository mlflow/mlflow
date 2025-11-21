import { describe, jest, beforeAll, beforeEach, it, expect, afterEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClientProvider, QueryClient } from '@databricks/web-shared/query-client';

import { useOverrideAssessment } from './useOverrideAssessment';
import { useTraceCachedActions } from './useTraceCachedActions';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment, ModelTraceInfoV3, ModelTraceLocation } from '../ModelTrace.types';
import type { CreateAssessmentPayload } from '../api';
import { ModelTraceExplorerUpdateTraceContextProvider } from '../contexts/UpdateTraceContext';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: jest.fn(),
  doesTraceSupportV4API: jest.fn(() => true),
}));

jest.mock('@databricks/web-shared/global-settings', () => ({
  getUser: jest.fn(() => 'test-user@databricks.com'),
  getOrgID: jest.fn(() => '123456'),
}));

describe('useOverrideAssessment', () => {
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
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false },
              },
            })
          }
        >
          <ModelTraceExplorerUpdateTraceContextProvider modelTraceInfo={mockTraceInfo} sqlWarehouseId="warehouse-123">
            {children}
          </ModelTraceExplorerUpdateTraceContextProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );

  const mockOldAssessment: Assessment = {
    assessment_id: 'assessment-old-123',
    assessment_name: 'Correctness',
    trace_id: 'trace-123',
    source: {
      source_type: 'LLM_JUDGE',
      source_id: 'databricks',
    },
    feedback: {
      value: 'no',
    },
    create_time: '2025-11-16T10:00:00.000Z',
    last_update_time: '2025-11-16T10:00:00.000Z',
    valid: true,
  } as Assessment;

  const mockNewAssessmentResponse = {
    assessment: {
      assessment_id: 'assessment-new-456',
      assessment_name: 'Correctness',
      trace_id: 'trace-123',
      source: {
        source_type: 'HUMAN',
        source_id: 'test-user@databricks.com',
      },
      feedback: {
        value: 'yes',
      },
      overrides: 'assessment-old-123',
      create_time: '2025-11-16T12:00:00.000Z',
      last_update_time: '2025-11-16T12:00:00.000Z',
      valid: true,
    } as Assessment,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    renderHook(() => useTraceCachedActions()).result.current.resetCache();
  });

  describe('when V4 API is disabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
      server.use(
        rest.post('*/ajax-api/*/mlflow/traces/trace-123/assessments', async (req, res, ctx) =>
          res(ctx.json(mockNewAssessmentResponse)),
        ),
      );
    });

    it('should override assessment using V3 API', async () => {
      const createSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.post('*/ajax-api/*/mlflow/traces/trace-123/assessments', async (req, res, ctx) => {
          createSpy(await req.json());
          return res(ctx.json(mockNewAssessmentResponse));
        }),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useOverrideAssessment({ traceId: 'trace-123', onSuccess }), { wrapper });

      result.current.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
        rationale: 'Human override',
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(createSpy).toHaveBeenCalled();

      const payload = createSpy.mock.calls[0][0] as CreateAssessmentPayload;
      expect(payload.assessment.overrides).toBe('assessment-old-123');
      expect(payload.assessment.source.source_type).toBe('HUMAN');
    });

    it('should not update cache when V4 API is disabled', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          overrideHook: useOverrideAssessment({ traceId: 'trace-123', onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.overrideHook.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(result.current.cacheHook.assessmentActions['trace-123']).toBeUndefined();
    });
  });

  describe('when V4 API is enabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(true);
      server.use(
        rest.post('*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments', async (req, res, ctx) =>
          res(ctx.json(mockNewAssessmentResponse)),
        ),
      );
    });

    it('should override assessment using V4 API', async () => {
      const createSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.post('*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments', async (req, res, ctx) => {
          createSpy(await req.json());
          return res(ctx.json(mockNewAssessmentResponse));
        }),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useOverrideAssessment({ traceId: 'trace-123', onSuccess }), { wrapper });

      result.current.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
        rationale: 'Human override',
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(createSpy).toHaveBeenCalled();
    });

    it('should add new assessment to cache and delete old assessment', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          overrideHook: useOverrideAssessment({ traceId: 'trace-123', onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.overrideHook.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());

      const actions = result.current.cacheHook.assessmentActions['trace-123'];
      expect(actions).toHaveLength(2);
      expect(actions[0]).toMatchObject({
        action: 'ADD',
        assessment: expect.objectContaining({
          assessment_id: 'assessment-new-456',
        }),
      });
      expect(actions[1]).toMatchObject({
        action: 'DELETE',
        assessment: expect.objectContaining({
          assessment_id: 'assessment-old-123',
        }),
      });
    });

    it('should populate overriddenAssessment field in new assessment', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          overrideHook: useOverrideAssessment({ traceId: 'trace-123', onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      result.current.overrideHook.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());

      const actions = result.current.cacheHook.assessmentActions['trace-123'];
      const addedAssessment = actions[0].assessment;
      expect(addedAssessment?.overriddenAssessment).toEqual(mockOldAssessment);
    });

    it('should include rationale in override', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(() => useOverrideAssessment({ traceId: 'trace-123', onSuccess }), { wrapper });

      result.current.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
        rationale: 'The agent actually performed correctly',
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    });

    it('should handle expectation overrides', async () => {
      server.resetHandlers();
      server.use(
        rest.post('*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments', async (req, res, ctx) =>
          res(
            ctx.json({
              assessment: {
                ...mockNewAssessmentResponse.assessment,
                expectation: { value: '{"key": "value"}' },
              },
            }),
          ),
        ),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useOverrideAssessment({ traceId: 'trace-123', onSuccess }), { wrapper });

      const oldExpectation = { ...mockOldAssessment, expectation: { value: '{"old": "value"}' } };
      result.current.overrideAssessmentMutation({
        oldAssessment: oldExpectation as Assessment,
        value: { expectation: { value: '{"key": "value"}' } },
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    });

    it('should call onSettled callback', async () => {
      const onSuccess = jest.fn();
      const onSettled = jest.fn();
      const { result } = renderHook(() => useOverrideAssessment({ traceId: 'trace-123', onSuccess, onSettled }), {
        wrapper,
      });

      result.current.overrideAssessmentMutation({
        oldAssessment: mockOldAssessment,
        value: { feedback: { value: 'yes' } },
      });

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(onSettled).toHaveBeenCalled();
    });
  });
});
