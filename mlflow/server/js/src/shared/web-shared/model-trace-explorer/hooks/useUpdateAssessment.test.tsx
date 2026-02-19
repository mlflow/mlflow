import { describe, jest, beforeAll, beforeEach, it, expect, afterEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClientProvider, QueryClient } from '@databricks/web-shared/query-client';

import { useTraceCachedActions } from './useTraceCachedActions';
import { useUpdateAssessment } from './useUpdateAssessment';
import { shouldUseTracesV4API } from '../FeatureUtils';
import type { Assessment, ModelTraceInfoV3, ModelTraceLocation } from '../ModelTrace.types';
import type { UpdateAssessmentPayload } from '../api';
import { ModelTraceExplorerUpdateTraceContextProvider } from '../contexts/UpdateTraceContext';

jest.mock('../FeatureUtils', () => ({
  shouldUseTracesV4API: jest.fn(),
  doesTraceSupportV4API: jest.fn(() => true),
}));

jest.mock('@databricks/web-shared/global-settings', () => ({
  getUser: jest.fn(() => 'test-user@databricks.com'),
  getOrgID: jest.fn(() => '123456'),
}));

describe('useUpdateAssessment', () => {
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
      source_id: 'test-user@databricks.com',
    },
    feedback: {
      value: 'yes',
    },
    create_time: '2025-11-16T10:00:00.000Z',
    last_update_time: '2025-11-16T10:00:00.000Z',
    valid: true,
  } as Assessment;

  const mockUpdatedAssessmentResponse = {
    assessment: {
      ...mockAssessment,
      feedback: {
        value: 'no',
      },
      last_update_time: '2025-11-16T12:00:00.000Z',
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
        rest.patch('*/ajax-api/*/mlflow/traces/trace-123/assessments/assessment-123', async (req, res, ctx) =>
          res(ctx.json(mockUpdatedAssessmentResponse)),
        ),
      );
    });

    it('should update assessment using V3 API', async () => {
      const updateSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.patch('*/ajax-api/*/mlflow/traces/trace-123/assessments/assessment-123', async (req, res, ctx) => {
          updateSpy(await req.json());
          return res(ctx.json(mockUpdatedAssessmentResponse));
        }),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useUpdateAssessment({ assessment: mockAssessment, onSuccess }), { wrapper });

      const payload = {
        assessment: {
          feedback: {
            value: 'no',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(updateSpy).toHaveBeenCalled();

      const receivedPayload = updateSpy.mock.calls[0][0] as UpdateAssessmentPayload;
      expect(receivedPayload.assessment.feedback?.value).toBe('no');
    });

    it('should not update cache when V4 API is disabled', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          updateHook: useUpdateAssessment({ assessment: mockAssessment, onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      const payload = {
        assessment: {
          feedback: {
            value: 'no',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateHook.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(result.current.cacheHook.assessmentActions['trace-123']).toBeUndefined();
    });
  });

  describe('when V4 API is enabled', () => {
    beforeEach(() => {
      jest.mocked(shouldUseTracesV4API).mockReturnValue(true);
      server.use(
        rest.patch(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) => res(ctx.json(mockUpdatedAssessmentResponse)),
        ),
      );
    });

    it('should update assessment using V4 API', async () => {
      const updateSpy = jest.fn();
      server.resetHandlers();
      server.use(
        rest.patch(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) => {
            updateSpy(await req.json());
            return res(ctx.json(mockUpdatedAssessmentResponse));
          },
        ),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useUpdateAssessment({ assessment: mockAssessment, onSuccess }), { wrapper });

      const payload = {
        assessment: {
          feedback: {
            value: 'no',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(updateSpy).toHaveBeenCalled();
    });

    it('should add update action to cache', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(
        () => ({
          updateHook: useUpdateAssessment({ assessment: mockAssessment, onSuccess }),
          cacheHook: useTraceCachedActions(),
        }),
        { wrapper },
      );

      const payload = {
        assessment: {
          feedback: {
            value: 'no',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateHook.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());

      const actions = result.current.cacheHook.assessmentActions['trace-123'];
      expect(actions).toHaveLength(1);
      expect(actions[0]).toMatchObject({
        action: 'ADD',
        assessment: expect.objectContaining({
          assessment_id: 'assessment-123',
          feedback: {
            value: 'no',
          },
        }),
      });
    });

    it('should update feedback value', async () => {
      const onSuccess = jest.fn();
      const { result } = renderHook(() => useUpdateAssessment({ assessment: mockAssessment, onSuccess }), { wrapper });

      const payload = {
        assessment: {
          feedback: {
            value: 'no',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    });

    it('should update expectation value', async () => {
      const expectationAssessment = {
        ...mockAssessment,
        expectation: { value: '{"old": "value"}' },
      } as Assessment;

      server.resetHandlers();
      server.use(
        rest.patch(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) =>
            res(
              ctx.json({
                assessment: {
                  ...expectationAssessment,
                  expectation: { value: '{"new": "value"}' },
                },
              }),
            ),
        ),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useUpdateAssessment({ assessment: expectationAssessment, onSuccess }), {
        wrapper,
      });

      const payload = {
        assessment: {
          expectation: {
            value: '{"new": "value"}',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    });

    it('should update rationale', async () => {
      server.resetHandlers();
      server.use(
        rest.patch(
          '*/ajax-api/4.0/mlflow/traces/my_catalog.my_schema/trace-123/assessments/assessment-123',
          async (req, res, ctx) =>
            res(
              ctx.json({
                assessment: {
                  ...mockUpdatedAssessmentResponse.assessment,
                  rationale: 'Updated reasoning',
                },
              }),
            ),
        ),
      );

      const onSuccess = jest.fn();
      const { result } = renderHook(() => useUpdateAssessment({ assessment: mockAssessment, onSuccess }), { wrapper });

      const payload = {
        assessment: {
          rationale: 'Updated reasoning',
        },
      } as UpdateAssessmentPayload;

      result.current.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
    });

    it('should call onSettled callback', async () => {
      const onSuccess = jest.fn();
      const onSettled = jest.fn();
      const { result } = renderHook(() => useUpdateAssessment({ assessment: mockAssessment, onSuccess, onSettled }), {
        wrapper,
      });

      const payload = {
        assessment: {
          feedback: {
            value: 'no',
          },
        },
      } as UpdateAssessmentPayload;

      result.current.updateAssessmentMutation(payload);

      await waitFor(() => expect(onSuccess).toHaveBeenCalled());
      expect(onSettled).toHaveBeenCalled();
    });
  });
});
