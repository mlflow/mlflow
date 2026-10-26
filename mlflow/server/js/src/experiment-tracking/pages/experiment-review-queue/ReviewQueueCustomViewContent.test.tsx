import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { GET_TRACE_BY_ID } from '@databricks/web-shared/model-trace-explorer';
import {
  CustomViewDefinitionProvider,
  getCustomViewAuthoringContext,
  useOptionalCustomViewDefinition,
  type CustomView,
} from '@databricks/web-shared/model-trace-explorer/custom-view';
import { getCustomViewSpecApplier } from '../../../shared/web-shared/model-trace-explorer/custom-view/assistant/customViewSpecApplier';
import {
  MOCK_TRACE_INFO_V3,
  MOCK_V3_TRACE,
} from '../../../shared/web-shared/model-trace-explorer/ModelTraceExplorer.test-utils';
import { setupServer } from '../../../shared/web-shared/test-utils/setup-msw';

import { REVIEW_QUEUE_TRACE_ASSESSMENTS_QUERY_KEY } from './hooks/useTraceAssessmentsQuery';
import { ReviewQueueCustomViewRenderer } from './ReviewQueueCustomViewContent';

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  __esModule: true,
  useAssistant: () => ({ canUseAssistant: false, isPanelOpen: false, openPanel: jest.fn() }),
}));

const savedView: CustomView = {
  id: 'saved-view-1',
  name: 'Reviewer view',
  label: 'Reviewer view',
  instruction: '',
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [{ id: 'root', component: 'Text', text: 'Saved view content' }],
      },
    },
  ],
  createdAtMs: 1,
};

const thumbsFeedbackView: CustomView = {
  ...savedView,
  id: 'thumbs-view',
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          { id: 'root', component: 'FeedbackThumbsUpDownButtons', label: 'Was this helpful?', name: 'Helpfulness' },
        ],
      },
    },
  ],
};

const SelectViewProbe = ({ selectViewRef }: { selectViewRef: { current?: (id: string) => void } }) => {
  selectViewRef.current = useOptionalCustomViewDefinition()?.selectView;
  return null;
};

const renderSavedView = (
  definitionProps: { onPersistView?: (view: CustomView) => Promise<void> },
  { view = savedView, queryClient = new QueryClient() }: { view?: CustomView; queryClient?: QueryClient } = {},
) => {
  const selectViewRef: { current?: (id: string) => void } = {};
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <CustomViewDefinitionProvider
            views={[view]}
            isLoaded
            onPersistView={definitionProps.onPersistView}
            canModifyPersistedViews={Boolean(definitionProps.onPersistView)}
          >
            <ReviewQueueCustomViewRenderer modelTrace={MOCK_V3_TRACE} enableFeedback />
            <SelectViewProbe selectViewRef={selectViewRef} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  act(() => selectViewRef.current?.(view.id));
};

describe('ReviewQueueCustomViewRenderer', () => {
  const { server } = setupServer();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders a saved view without publishing Assistant authoring hooks from a load-only definition', async () => {
    renderSavedView({});

    expect(await screen.findByText('Saved view content')).toBeInTheDocument();
    expect(getCustomViewAuthoringContext()).toBeNull();
    expect(getCustomViewSpecApplier()).toBeNull();
  });

  it('publishes Assistant authoring hooks only when the definition can persist views', async () => {
    renderSavedView({ onPersistView: () => Promise.resolve() });

    expect(await screen.findByText('Saved view content')).toBeInTheDocument();
    expect(getCustomViewAuthoringContext()).not.toBeNull();
    expect(getCustomViewSpecApplier()).not.toBeNull();
  });

  it('refetches the trace and prior answers focused review shows after feedback is saved in the view', async () => {
    server.use(
      rest.post('*/ajax-api/*/mlflow/traces/*/assessments', async (req, res, ctx) =>
        res(ctx.json({ assessment: { ...(await req.json()).assessment, assessment_id: 'a-new' } })),
      ),
    );
    const traceId = MOCK_TRACE_INFO_V3.trace_id;
    const traceKey = [GET_TRACE_BY_ID, traceId];
    const priorAnswersKey = [REVIEW_QUEUE_TRACE_ASSESSMENTS_QUERY_KEY, traceId, 'reviewer'];
    const queryClient = new QueryClient();
    queryClient.setQueryData(traceKey, MOCK_V3_TRACE);
    queryClient.setQueryData(priorAnswersKey, []);
    renderSavedView({}, { view: thumbsFeedbackView, queryClient });

    await userEvent.click(await screen.findByRole('button', { name: 'Thumbs up' }));

    await waitFor(() => expect(queryClient.getQueryState(traceKey)?.isInvalidated).toBe(true));
    expect(queryClient.getQueryState(priorAnswersKey)?.isInvalidated).toBe(true);
  });
});
